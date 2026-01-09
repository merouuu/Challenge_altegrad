import argparse
import os
import json
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
parser = argparse.ArgumentParser(description="Graph Transformer Training Script")
parser.add_argument('--env', type=str, default='local', choices=['local', 'colab'], 
                    help="Environment: 'local' or 'colab'")
parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate")
args = parser.parse_args()

base_path = "/content/drive/MyDrive/data" if args.env == 'colab' else "data"
TRAIN_GRAPHS = f"{base_path}/train_graphs.pkl"
VAL_GRAPHS   = f"{base_path}/validation_graphs.pkl"
TRAIN_EMB_CSV = f"{base_path}/train_embeddings.csv"
VAL_EMB_CSV   = f"{base_path}/validation_embeddings.csv"

# --- FIX: Create the directory first ---
OUTPUT_DIR = f"{base_path}/GT"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created directory: {OUTPUT_DIR}")

MODEL_SAVE_PATH = f"{OUTPUT_DIR}/transformer_model.pt"
LOGS_SAVE_PATH = f"{OUTPUT_DIR}/training_logs.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# FEATURE ENCODERS
# =========================================================
class AtomEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # Based on ALTEGRAD dataset specs (9 features)
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
        # Based on ALTEGRAD dataset specs (3 features)
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
                    edge_dim=hidden, # Incorporates bond features into attention
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

        # Global Mean Pooling (more stable than add_pool for embeddings)
        g = global_mean_pool(x, batch.batch)
        
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        return g

# =========================================================
# TRAINING & EVAL UTILS
# =========================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_count = 0.0, 0
    
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        # Forward pass
        mol_vec = model(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)

        # MSE Loss (cosine distance proxy because vectors are normalized)
        loss = F.mse_loss(mol_vec, txt_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * graphs.num_graphs
        total_count += graphs.num_graphs

    return total_loss / total_count

@torch.no_grad()
def eval_retrieval(data_path, emb_dict, model, device, compute_loss=True):
    """
    Évalue le modèle sur le validation set.
    
    Args:
        data_path: Chemin vers les graphes de validation
        emb_dict: Dictionnaire des embeddings de texte
        model: Modèle à évaluer
        device: Device (cuda/cpu)
        compute_loss: Si True, calcule aussi la validation loss (MSE)
    
    Returns:
        Dictionnaire avec les métriques de retrieval et optionnellement la loss
    """
    model.eval()
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    total_val_loss, total_count = 0.0, 0
    
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        
        mol_vec = model(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)
        
        # Calcul de la validation loss si demandé (MSE pour train_gt.py)
        if compute_loss:
            val_loss = F.mse_loss(mol_vec, txt_vec)
            total_val_loss += val_loss.item() * graphs.num_graphs
            total_count += graphs.num_graphs
        
        all_mol.append(mol_vec)
        all_txt.append(txt_vec)
        
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    # Cosine Similarity: Matrix Multiplication
    sims = all_txt @ all_mol.t()
    
    # Calculate Ranks
    ranks = sims.argsort(dim=-1, descending=True)
    N = all_txt.size(0)
    correct_indices = torch.arange(N, device=device).unsqueeze(1)
    
    # Find where the correct answer is in the ranked list
    pos = (ranks == correct_indices).nonzero()[:, 1] + 1
    
    mrr = (1.0 / pos.float()).mean().item()
    r1 = (pos <= 1).float().mean().item()
    r5 = (pos <= 5).float().mean().item()
    r10 = (pos <= 10).float().mean().item()

    results = {"MRR": mrr, "R@1": r1, "R@5": r5, "R@10": r10}
    
    # Ajouter la validation loss si calculée
    if compute_loss:
        results["val_loss"] = total_val_loss / total_count if total_count > 0 else 0.0

    return results

# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Running on: {DEVICE}")
    print(f"Data Path: {base_path}")

    # Load Embeddings
    print("Loading embeddings...")
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    
    # Validate Data Availability
    if not os.path.exists(TRAIN_GRAPHS):
        raise FileNotFoundError(f"Graphs not found at {TRAIN_GRAPHS}. Run prepare_graph_data.py first.")

    # Setup DataLoaders
    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Determine Embedding Dimension from data
    emb_dim = len(next(iter(train_emb.values())))
    
    # Initialize Model
    model = MolGraphTransformer(hidden=128, out_dim=emb_dim, layers=4, heads=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training Loop
    best_mrr = 0.0
    training_logs = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr
        },
        "epochs": []
    }
    
    for ep in range(args.epochs):
        loss = train_epoch(model, train_dl, optimizer, DEVICE)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation (only if file exists)
        val_metrics = {}
        if os.path.exists(VAL_GRAPHS) and os.path.exists(VAL_EMB_CSV):
            val_emb = load_id2emb(VAL_EMB_CSV)
            # Calculer les métriques de retrieval ET la validation loss
            val_metrics = eval_retrieval(VAL_GRAPHS, val_emb, model, DEVICE, compute_loss=True)
            
            # Save best model based on MRR
            if val_metrics["MRR"] > best_mrr:
                best_mrr = val_metrics["MRR"]
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  [New Best Model Saved] MRR: {best_mrr:.4f}")

        # Enregistrer les logs
        epoch_log = {
            "epoch": ep + 1,
            "train_loss": float(loss),
            "learning_rate": float(current_lr),
            "val_loss": float(val_metrics.get("val_loss", 0.0)),
            "val_mrr": float(val_metrics.get("MRR", 0.0)),
            "val_r1": float(val_metrics.get("R@1", 0.0)),
            "val_r5": float(val_metrics.get("R@5", 0.0)),
            "val_r10": float(val_metrics.get("R@10", 0.0))
        }
        training_logs["epochs"].append(epoch_log)
        
        # Sauvegarder les logs à chaque epoch
        with open(LOGS_SAVE_PATH, 'w') as f:
            json.dump(training_logs, f, indent=2)

        val_str = f"Loss: {val_metrics.get('val_loss', 0.0):.5f}, " if val_metrics.get('val_loss') else ""
        val_str += f"MRR: {val_metrics.get('MRR', 0.0):.4f}, R@1: {val_metrics.get('R@1', 0.0):.4f}"
        print(f"Epoch {ep+1:02d} | Train Loss: {loss:.5f} | Val: {val_str}")

    training_logs["best_mrr"] = float(best_mrr)
    with open(LOGS_SAVE_PATH, 'w') as f:
        json.dump(training_logs, f, indent=2)
    
    print(f"\nTraining Complete. Best Model saved to {MODEL_SAVE_PATH}")
    print(f"Training logs saved to {LOGS_SAVE_PATH}")

if __name__ == "__main__":
    main()