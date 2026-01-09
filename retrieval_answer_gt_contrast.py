import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool

from data_utils import (
    load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
)

# Configuration des arguments
parser = argparse.ArgumentParser(description="Generate Submission for Contrastive GT Model")
parser.add_argument('--env', type=str, default='local', choices=['local', 'colab'], 
                    help="Define environment: 'local' or 'colab'")
args = parser.parse_args()

# =========================================================
# CONFIGURATION
# =========================================================
base_path = "/content/drive/MyDrive/data" if args.env == 'colab' else "data"
TRAIN_GRAPHS = f"{base_path}/train_graphs.pkl"
TEST_GRAPHS  = f"{base_path}/test_graphs.pkl"
TRAIN_EMB_CSV = f"{base_path}/GCN/train_embeddings.csv"

# PATHS SPECIFIC TO CONTRASTIVE MODEL
# We look for the model in the GT_Contrast folder
MODEL_PATH = f"{base_path}/GT_Contrast/contrastive_model.pt"
OUTPUT_CSV = f"{base_path}/GT_Contrast/submission_contrastive.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# MODEL DEFINITION (Must match training exactly)
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
        g = F.normalize(g, dim=-1)
        return g

# =========================================================
# RETRIEVAL LOGIC
# =========================================================
@torch.no_grad()
def retrieve_descriptions(model, train_data, test_data, train_emb_dict, device, output_csv):
    print("--- Starting Retrieval Process (Contrastive Model) ---")
    
    # 1. Load Ground Truth Descriptions
    print("Loading train descriptions...")
    train_id2desc = load_descriptions_from_graphs(train_data)
    
    # 2. Prepare Text Embeddings (Targets)
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[id_] for id_ in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)
    
    print(f"Target Database size: {len(train_ids)}")
    
    # 3. Load Test Graphs
    test_ds = PreprocessedGraphDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    print(f"Test Queries size: {len(test_ds)}")
    
    # 4. Encode Test Graphs
    model.eval()
    test_mol_embs = []
    print("Encoding test graphs...")
    for graphs in test_dl:
        graphs = graphs.to(device)
        mol_emb = model(graphs)
        test_mol_embs.append(mol_emb)
    
    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    
    # 5. Calculate Similarity (Dot Product)
    print("Calculating similarity matrix...")
    similarities = test_mol_embs @ train_embs.t()
    
    # 6. Retrieve Best Matches
    most_similar_indices = similarities.argmax(dim=-1).cpu()
    test_ids_ordered = test_ds.ids 

    results = []
    for i, test_id in enumerate(test_ids_ordered):
        train_idx = most_similar_indices[i].item()
        retrieved_train_id = train_ids[train_idx]
        retrieved_desc = train_id2desc[retrieved_train_id]
        
        results.append({
            'ID': test_id,
            'description': retrieved_desc
        })
        
        if i < 3: # Print first few examples
            print(f"\n[Example] Test ID {test_id} -> Train ID {retrieved_train_id}")
            print(f"Caption: {retrieved_desc[:100]}...")
    
    # 7. Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Submission saved to : {output_csv}")
    print(f"{'='*80}")
    
    return results_df

def main():
    print(f"Environment : {args.env}")
    print(f"Device      : {DEVICE}")
    print(f"Model Path  : {MODEL_PATH}")
    
    # Check files
    if not os.path.exists(TEST_GRAPHS):
        print(f"Error: Test graphs not found at {TEST_GRAPHS}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        print("Please run train_gt_contrast.py first.")
        return

    # Load Embeddings to get dimension
    print("Loading train embeddings...")
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    emb_dim = len(next(iter(train_emb.values())))
    
    # Instantiate Model
    # Hyperparameters must match train_gt_contrast.py
    model = MolGraphTransformer(hidden=128, out_dim=emb_dim, layers=4, heads=4).to(DEVICE)
    
    # Load Weights
    print(f"Loading weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # Run Retrieval
    retrieve_descriptions(
        model=model,
        train_data=TRAIN_GRAPHS,
        test_data=TEST_GRAPHS,
        train_emb_dict=train_emb,
        device=DEVICE,
        output_csv=OUTPUT_CSV
    )

if __name__ == "__main__":
    main()