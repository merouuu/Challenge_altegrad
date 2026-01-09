import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool, LayerNorm
from torch_geometric.utils import scatter

from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn
)

# =========================================================
# CONFIG & ARGS
# =========================================================
parser = argparse.ArgumentParser(description="Improved Graph Transformer + Contrastive Loss")
parser.add_argument('--env', type=str, default='local', choices=['local', 'colab'], 
                    help="Environment: 'local' or 'colab'")
parser.add_argument('--epochs', type=int, default=20, help="Number of epochs (increased for better convergence)")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser.add_argument('--lr', type=float, default=0.0003, help="Learning rate (balanced for improved architecture)")
parser.add_argument('--temp', type=float, default=0.07, help="Temperature for Contrastive Loss (lower = sharper distribution)")
parser.add_argument('--hidden', type=int, default=128, help="Hidden dimension")
parser.add_argument('--layers', type=int, default=4, help="Number of transformer layers")
parser.add_argument('--heads', type=int, default=4, help="Number of attention heads")
args = parser.parse_args()

base_path = "/content/drive/MyDrive/data" if args.env == 'colab' else "data"
TRAIN_GRAPHS = f"{base_path}/train_graphs.pkl"
VAL_GRAPHS   = f"{base_path}/validation_graphs.pkl"
TRAIN_EMB_CSV = f"{base_path}/train_embeddings.csv"
VAL_EMB_CSV   = f"{base_path}/validation_embeddings.csv"

OUTPUT_DIR = f"{base_path}/GT_Contrast"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created directory: {OUTPUT_DIR}")

MODEL_SAVE_PATH = f"{OUTPUT_DIR}/contrastive_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# LOSS FUNCTION: Contrastive (CLIP-style)
# =========================================================
def contrastive_loss(mol_emb, txt_emb, temperature=0.1):
    """
    Calcule la perte contrastive (InfoNCE).
    Maximise la similarité entre la molécule i et le texte i,
    tout en minimisant la similarité avec tous les autres textes j != i.
    """
    mol_emb = F.normalize(mol_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    
    logits = (mol_emb @ txt_emb.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    
    return (loss_i + loss_t) / 2.0


# =========================================================
# IMPROVED FEATURE ENCODERS
# =========================================================
class ImprovedAtomEncoder(nn.Module):
    """
    Encodage amélioré avec pondération apprise des features atomiques.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(120, emb_dim), # atomic_num
            nn.Embedding(10, emb_dim),  # chirality
            nn.Embedding(12, emb_dim),  # degree
            nn.Embedding(12, emb_dim),  # formal charge
            nn.Embedding(10, emb_dim),  # num_hs
            nn.Embedding(6, emb_dim),   # radical
            nn.Embedding(10, emb_dim),  # hybridization
            nn.Embedding(2, emb_dim),   # is_aromatic
            nn.Embedding(2, emb_dim)    # is_in_ring
        ])
        # Pondération apprise pour chaque feature
        self.feature_weights = nn.Parameter(torch.ones(len(self.embeddings)))
        # Projection pour combiner les features
        self.combine = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU()
        )

    def forward(self, x):
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            # Pondération apprise + embedding
            weighted_emb = emb(x[:, i]) * torch.sigmoid(self.feature_weights[i])
            embeddings.append(weighted_emb)
        
        # Somme pondérée
        out = sum(embeddings)
        # Projection avec normalisation
        out = self.combine(out)
        return out


class ImprovedBondEncoder(nn.Module):
    """
    Encodage amélioré des features de bonds avec pondération.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(20, emb_dim),  # bond_type
            nn.Embedding(10, emb_dim),  # stereo
            nn.Embedding(2, emb_dim)    # is_conjugated
        ])
        # Pondération apprise
        self.feature_weights = nn.Parameter(torch.ones(len(self.embeddings)))
        # Projection
        self.combine = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU()
        )

    def forward(self, edge_attr):
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            weighted_emb = emb(edge_attr[:, i]) * torch.sigmoid(self.feature_weights[i])
            embeddings.append(weighted_emb)
        
        out = sum(embeddings)
        out = self.combine(out)
        return out


# =========================================================
# ATTENTION POOLING
# =========================================================
class AttentionPooling(nn.Module):
    """
    Pooling attentionné au lieu de simple mean pooling.
    Permet au modèle de se concentrer sur les atomes les plus importants.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, batch):
        # x: [num_nodes, hidden_dim]
        # batch: [num_nodes] - indique à quel graphe appartient chaque nœud
        
        # Calcul des poids d'attention
        attn_logits = self.attention(x)  # [num_nodes, 1]
        
        # Softmax par graphe (normalisation par batch)
        attn_weights = scatter(
            attn_logits.exp(), 
            batch, 
            dim=0, 
            reduce='sum'
        )  # [num_graphs, 1]
        
        # Normalisation: attn_weights[i] = exp(score) / sum(exp(scores) for graph i)
        attn_weights = attn_logits.exp() / (scatter(attn_logits.exp(), batch, dim=0, reduce='sum')[batch] + 1e-8)
        
        # Pooling attentionné: somme pondérée par graphe
        weighted_x = x * attn_weights
        graph_embeddings = scatter(weighted_x, batch, dim=0, reduce='sum')
        
        return graph_embeddings


# =========================================================
# IMPROVED GRAPH TRANSFORMER MODEL
# =========================================================
class ImprovedMolGraphTransformer(nn.Module):
    """
    Graph Transformer amélioré avec:
    - Skip connections
    - LayerNorm
    - Attention pooling
    - Positional encoding
    - Projection plus profonde
    """
    def __init__(self, hidden=128, out_dim=256, layers=4, heads=4, dropout=0.1):
        super().__init__()
        
        self.atom_encoder = ImprovedAtomEncoder(hidden)
        self.bond_encoder = ImprovedBondEncoder(hidden)
        
        # Positional encoding basé sur le degré du nœud
        # Le degré est une feature importante pour la structure moléculaire
        self.pos_encoder = nn.Embedding(12, hidden)  # max degree = 11
        
        # Transformer layers avec skip connections et LayerNorm
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for _ in range(layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden,
                    out_channels=hidden // heads,
                    heads=heads,
                    edge_dim=hidden,
                    dropout=dropout,
                    concat=True  # Important: concat pour avoir hidden_dim en sortie
                )
            )
            self.layer_norms.append(LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Attention pooling au lieu de mean pooling
        self.attention_pool = AttentionPooling(hidden)
        
        # Projection plus profonde avec normalisation
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, batch):
        # Encodage des atomes avec pondération
        x = self.atom_encoder(batch.x)
        
        # Positional encoding basé sur le degré (feature index 2)
        degree = batch.x[:, 2].long().clamp(0, 11)  # Clamp pour sécurité
        pos_emb = self.pos_encoder(degree)
        x = x + pos_emb  # Ajout du positional encoding
        
        # Encodage des bonds
        edge_attr = self.bond_encoder(batch.edge_attr)
        
        # Transformer layers avec skip connections
        for i, (conv, ln, dropout) in enumerate(zip(self.convs, self.layer_norms, self.dropouts)):
            residual = x
            x = conv(x, batch.edge_index, edge_attr)
            x = ln(x)
            x = F.relu(x)
            x = dropout(x)
            
            # Skip connection (si dimensions compatibles)
            if x.shape == residual.shape:
                x = x + residual

        # Attention pooling au lieu de mean pooling
        g = self.attention_pool(x, batch.batch)
        
        # Projection finale
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        return g


# =========================================================
# TRAINING LOOP
# =========================================================
def train_epoch(model, loader, optimizer, device, temp):
    model.train()
    total_loss, total_count = 0.0, 0
    
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = model(graphs)
        loss = contrastive_loss(mol_vec, text_emb, temperature=temp)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / total_count


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, model, device):
    model.eval()
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        
        all_mol.append(model(graphs))
        all_txt.append(F.normalize(text_emb, dim=-1))
        
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    sims = all_txt @ all_mol.t()
    
    ranks = sims.argsort(dim=-1, descending=True)
    N = all_txt.size(0)
    correct_indices = torch.arange(N, device=device).unsqueeze(1)
    
    pos = (ranks == correct_indices).nonzero()[:, 1] + 1
    
    mrr = (1.0 / pos.float()).mean().item()
    r1 = (pos <= 1).float().mean().item()
    r5 = (pos <= 5).float().mean().item()
    r10 = (pos <= 10).float().mean().item()

    return {"MRR": mrr, "R@1": r1, "R@5": r5, "R@10": r10}


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Running on: {DEVICE}")
    print(f"Data Path: {base_path}")
    print(f"Loss: Contrastive (Temp={args.temp})")
    print(f"Architecture: Improved Transformer (hidden={args.hidden}, layers={args.layers}, heads={args.heads})")
    
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    
    if not os.path.exists(TRAIN_GRAPHS):
        raise FileNotFoundError(f"Graphs not found at {TRAIN_GRAPHS}")

    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    # Drop_last=True important pour la loss contrastive
    train_dl = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        drop_last=True
    )

    emb_dim = len(next(iter(train_emb.values())))
    
    model = ImprovedMolGraphTransformer(
        hidden=args.hidden, 
        out_dim=emb_dim, 
        layers=args.layers, 
        heads=args.heads
    ).to(DEVICE)
    
    # Compter les paramètres
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler - réduit le LR si pas d'amélioration pendant 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    best_mrr = 0.0
    for ep in range(args.epochs):
        train_loss = train_epoch(model, train_dl, optimizer, DEVICE, args.temp)
        
        val_metrics = {}
        if os.path.exists(VAL_GRAPHS) and os.path.exists(VAL_EMB_CSV):
            val_emb = load_id2emb(VAL_EMB_CSV)
            val_metrics = eval_retrieval(VAL_GRAPHS, val_emb, model, DEVICE)
            
            # Scheduler step basé sur MRR
            scheduler.step(val_metrics["MRR"])
            
            if val_metrics["MRR"] > best_mrr:
                best_mrr = val_metrics["MRR"]
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  [New Best Model Saved] MRR: {best_mrr:.4f}")

        print(f"Epoch {ep+1:02d}/{args.epochs} | Loss: {train_loss:.4f} | Val: {val_metrics}")

    print(f"\nTraining Complete. Best Model saved to {MODEL_SAVE_PATH}")
    print(f"Best MRR: {best_mrr:.4f}")

if __name__ == "__main__":
    main()