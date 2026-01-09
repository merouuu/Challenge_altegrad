import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool

from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn
)

# =========================================================
# CONFIG & ARGS
# =========================================================
parser = argparse.ArgumentParser(description="Graph Transformer + Contrastive Loss")
parser.add_argument('--env', type=str, default='local', choices=['local', 'colab'], 
                    help="Environment: 'local' or 'colab'")
parser.add_argument('--epochs', type=int, default=15, help="Number of epochs (More needed for contrastive)")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate (Lower is safer for Transformer)")
parser.add_argument('--temp', type=float, default=0.1, help="Temperature for Contrastive Loss")
args = parser.parse_args()

base_path = "/content/drive/MyDrive/data" if args.env == 'colab' else "data"
TRAIN_GRAPHS = f"{base_path}/train_graphs.pkl"
VAL_GRAPHS   = f"{base_path}/validation_graphs.pkl"
TRAIN_EMB_CSV = f"{base_path}/train_embeddings.csv"
VAL_EMB_CSV   = f"{base_path}/validation_embeddings.csv"

# --- OUTPUT DIRECTORY ---
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
    # 1. Normalisation (Essentiel pour le produit scalaire / cosinus)
    mol_emb = F.normalize(mol_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    
    # 2. Calcul de la matrice de similarité (Batch x Batch)
    # logits[i, j] = cos_sim(mol_i, txt_j) / temp
    logits = (mol_emb @ txt_emb.t()) / temperature
    
    # 3. Les labels sont la diagonale (0, 1, 2, ..., BatchSize-1)
    labels = torch.arange(logits.size(0), device=logits.device)
    
    # 4. Loss symétrique (Image->Texte et Texte->Image)
    loss_i = F.cross_entropy(logits, labels)      # Pour chaque mol, quel est le bon texte ?
    loss_t = F.cross_entropy(logits.t(), labels)  # Pour chaque texte, quelle est la bonne mol ?
    
    return (loss_i + loss_t) / 2.0


# =========================================================
# FEATURE ENCODERS
# =========================================================
class AtomEncoder(nn.Module):
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

    def forward(self, x):
        out = 0
        for i, emb in enumerate(self.embeddings):
            out += emb(x[:, i])
        return out

class BondEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(20, emb_dim),  # bond_type
            nn.Embedding(10, emb_dim),  # stereo
            nn.Embedding(2, emb_dim)    # is_conjugated
        ])

    def forward(self, edge_attr):
        out = 0
        for i, emb in enumerate(self.embeddings):
            out += emb(edge_attr[:, i])
        return out

# =========================================================
# GRAPH TRANSFORMER MODEL
# =========================================================
class MolGraphTransformer(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=4, heads=4, dropout=0.1):
        super().__init__()
        
        self.atom_encoder = AtomEncoder(hidden)
        self.bond_encoder = BondEncoder(hidden)

        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden,
                    out_channels=hidden // heads,
                    heads=heads,
                    edge_dim=hidden,
                    dropout=dropout
                )
            )

        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, batch):
        x = self.atom_encoder(batch.x)
        edge_attr = self.bond_encoder(batch.edge_attr)

        for conv in self.convs:
            x = conv(x, batch.edge_index, edge_attr)
            x = F.relu(x)

        g = global_mean_pool(x, batch.batch)
        
        g = self.proj(g)
        # Note: On ne normalise PAS ici si on utilise contrastive_loss qui le fait en interne,
        # mais le faire deux fois n'est pas grave. Pour la cohérence avec l'inférence, on le laisse.
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

        # Forward pass
        mol_vec = model(graphs)
        # Pas besoin de normaliser txt_vec ici, c'est fait dans la loss
        
        # --- NEW: Contrastive Loss ---
        loss = contrastive_loss(mol_vec, text_emb, temperature=temp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # On multiplie par bs car cross_entropy fait une moyenne par défaut
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
    
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    
    if not os.path.exists(TRAIN_GRAPHS):
        raise FileNotFoundError(f"Graphs not found at {TRAIN_GRAPHS}")

    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    # Drop_last=True est important pour la loss contrastive pour éviter les batchs trop petits instables
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    emb_dim = len(next(iter(train_emb.values())))
    
    model = MolGraphTransformer(hidden=128, out_dim=emb_dim, layers=4, heads=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_mrr = 0.0
    for ep in range(args.epochs):
        train_loss = train_epoch(model, train_dl, optimizer, DEVICE, args.temp)
        
        val_metrics = {}
        if os.path.exists(VAL_GRAPHS) and os.path.exists(VAL_EMB_CSV):
            val_emb = load_id2emb(VAL_EMB_CSV)
            val_metrics = eval_retrieval(VAL_GRAPHS, val_emb, model, DEVICE)
            
            if val_metrics["MRR"] > best_mrr:
                best_mrr = val_metrics["MRR"]
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  [New Best Model Saved] MRR: {best_mrr:.4f}")

        # Note: La loss contrastive a des valeurs plus élevées que la MSE (~2.0 à 4.0 généralement)
        print(f"Epoch {ep+1:02d} | Loss: {train_loss:.4f} | Val: {val_metrics}")

    print(f"\nTraining Complete. Best Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()