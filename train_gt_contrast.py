import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
from torch.utils.data import DataLoader, Sampler
from torch_geometric.nn import TransformerConv, global_mean_pool, LayerNorm
from torch_geometric.utils import scatter, softmax

# Imports pour évaluation texte (BLEU + BERTScore)
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
except:
    sentence_bleu = None

try:
    from bert_score import score as bert_score
except:
    bert_score = None

from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn
)

# =========================================================
# TEXT LOADING
# =========================================================
def load_id2text(csv_path, id_col="id", text_col="description"):
    """
    Charge un dictionnaire ID -> texte depuis un CSV.
    """
    if not os.path.exists(csv_path):
        print(f"⚠️  Fichier {csv_path} non trouvé, textes non disponibles")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        # Chercher les colonnes (cas-insensitif)
        cols = {c.lower(): c for c in df.columns}
        id_col_actual = cols.get(id_col.lower(), id_col)
        text_col_actual = cols.get(text_col.lower(), text_col)
        
        if id_col_actual not in df.columns or text_col_actual not in df.columns:
            print(f"⚠️  Colonnes {id_col}/{text_col} non trouvées dans {csv_path}")
            print(f"   Colonnes disponibles: {list(df.columns)}")
            return {}
        
        return dict(zip(df[id_col_actual].astype(str), df[text_col_actual].astype(str)))
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {csv_path}: {e}")
        return {}

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
parser.add_argument('--resume_from', type=str, default=None, help="Path to checkpoint to resume training from (e.g., 'GT_Contrast/contrastive_model.pt')")
parser.add_argument('--start_epoch', type=int, default=0, help="Starting epoch number (useful when resuming, will be auto-detected from logs if available)")
parser.add_argument('--run_id', type=str, default=None, help="Unique run ID for grid search (saves to GT_Contrast/run_{run_id}/)")
parser.add_argument('--hard_negatives', action='store_true', help="Enable Hard Negative Mining with semantics-aware batching")
parser.add_argument('--hard_ratio', type=float, default=0.5, help="Ratio of hard negatives in batch (0.0-1.0, default 0.5 = 50%%)")
parser.add_argument('--hardness_k', type=int, default=100, help="Number of nearest neighbors to consider for hard negatives")
parser.add_argument('--curriculum_epoch', type=int, default=0, help="Start hard negative mining after this epoch (0 = from start, >0 = curriculum learning)")
parser.add_argument('--temp_schedule', action='store_true', help="Use temperature schedule (starts at 0.1, decays to final temp)")
parser.add_argument('--use_augmentation', action='store_true', help="Enable graph augmentation (edge/node dropout)")
parser.add_argument('--eval_bleu_bert', action='store_true', help="Evaluate with BLEU-4 + BERTScore")
parser.add_argument('--rerank_topk', type=int, default=1, help="Top-k candidates to evaluate for reranking")
args = parser.parse_args()

base_path = "/content/drive/MyDrive/data" if args.env == 'colab' else "data"
TRAIN_GRAPHS = f"{base_path}/train_graphs.pkl"
VAL_GRAPHS   = f"{base_path}/validation_graphs.pkl"
TRAIN_EMB_CSV = f"{base_path}/train_embeddings.csv"
VAL_EMB_CSV   = f"{base_path}/validation_embeddings.csv"

# Gestion du run_id pour grid search
if args.run_id:
    OUTPUT_DIR = f"{base_path}/GT_Contrast/run_{args.run_id}"
else:
    OUTPUT_DIR = f"{base_path}/GT_Contrast"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created directory: {OUTPUT_DIR}")

MODEL_SAVE_PATH = f"{OUTPUT_DIR}/contrastive_model.pt"
CHECKPOINT_SAVE_PATH = f"{OUTPUT_DIR}/checkpoint.pt"  # Checkpoint complet avec optimizer/scheduler
LOGS_SAVE_PATH = f"{OUTPUT_DIR}/training_logs.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# HARD NEGATIVE SAMPLER
# =========================================================
class HardNegativeSampler(Sampler):
    """
    Semantics-Aware Batch Sampler pour Hard Negative Mining.
    
    Crée des batches où les molécules sont sémantiquement proches,
    forçant le modèle à apprendre des distinctions fines.
    """
    def __init__(self, text_embeddings_dict, batch_size, hard_ratio=0.5, hardness_k=100, dataset=None):
        """
        Args:
            text_embeddings_dict (dict): Dictionnaire ID -> embedding BERT
            batch_size (int): Taille du batch
            hard_ratio (float): Pourcentage du batch composé de hard negatives (0.0-1.0)
            hardness_k (int): Nombre de voisins les plus proches à considérer
            dataset: (Optional) Le PreprocessedGraphDataset pour récupérer l'ordre réel des IDs
        """
        self.batch_size = batch_size
        self.hard_ratio = hard_ratio
        self.hardness_k = hardness_k
        
        # CRITICAL FIX: Construire les embeddings dans l'ordre du dataset, pas ordre trié
        # Cela garantit que neighbors[idx] correspond à train_ds[idx]
        if dataset is not None and hasattr(dataset, 'graph_ids'):
            # Si le dataset expose ses IDs dans l'ordre réel
            self.ids = dataset.graph_ids
            print(f"   ✅ Utilisation de l'ordre réel du dataset (première ID: {self.ids[0]})")
        else:
            # Fallback: espérer que l'ordre des clés du dict correspond au dataset
            self.ids = list(text_embeddings_dict.keys())
            print(f"   ⚠️  ATTENTION: Utilisation de l'ordre lexicographique des IDs!")
            print(f"       Assurez-vous que PreprocessedGraphDataset ordonne ses graphes de la même façon.")
        
        embs_list = [text_embeddings_dict[id_] for id_ in self.ids]
        # Conversion optimisée numpy -> tensor
        embs = torch.from_numpy(np.array(embs_list, dtype=np.float32))
        
        self.num_samples = len(embs)
        self.num_batches = self.num_samples // batch_size
        
        # Calcul des Hard Negatives (pré-computation)
        print(f"\n🧲 Pré-calcul des Hard Negatives (Similarity Matrix)...")
        print(f"   Dataset size: {self.num_samples}")
        print(f"   Hard ratio: {hard_ratio*100:.0f}%, K neighbors: {hardness_k}")
        
        # Normalisation pour cosine similarity
        embs = F.normalize(embs, dim=1)
        
        # Calcul des K voisins les plus proches (par chunks pour gérer la RAM)
        self.neighbors = []
        chunk_size = 1000
        
        for i in range(0, len(embs), chunk_size):
            end = min(i + chunk_size, len(embs))
            # Similarité du chunk courant avec tout le dataset
            sims = embs[i:end] @ embs.t()
            
            # Mettre à -inf la diagonale pour ne pas se sélectionner soi-même
            for j in range(sims.size(0)):
                sims[j, i+j] = -float('inf')
            
            # Top-k plus proches voisins
            k = min(hardness_k, self.num_samples - 1)
            _, indices = sims.topk(k, dim=1)
            self.neighbors.extend(indices.tolist())
            
            if (i // chunk_size) % 10 == 0:
                print(f"   Processed {min(i+chunk_size, len(embs))}/{len(embs)} samples...")
        
        print("✅ Hard Negatives indexés.\n")
    
    def __iter__(self):
        # Mélanger les indices de départ
        all_indices = np.arange(self.num_samples)
        np.random.shuffle(all_indices)
        
        n_hard = int(self.batch_size * self.hard_ratio)
        n_rand = self.batch_size - n_hard
        
        used_indices = set()
        batches = []
        
        for idx in all_indices:
            if idx in used_indices:
                continue
            
            batch = [idx]
            used_indices.add(idx)
            
            # 1. Ajouter des Hard Negatives (voisins proches)
            if n_hard > 0 and len(self.neighbors[idx]) > 0:
                candidates = self.neighbors[idx].copy()
                np.random.shuffle(candidates)
                
                count_h = 0
                for neighbor in candidates:
                    if count_h >= n_hard - 1:  # -1 car on a déjà ajouté idx
                        break
                    if neighbor not in used_indices:
                        batch.append(neighbor)
                        used_indices.add(neighbor)
                        count_h += 1
            
            # 2. Compléter avec des Random Negatives
            attempts = 0
            max_attempts = self.num_samples * 2
            while len(batch) < self.batch_size and attempts < max_attempts:
                rand_idx = np.random.randint(0, self.num_samples)
                if rand_idx not in used_indices:
                    batch.append(rand_idx)
                    used_indices.add(rand_idx)
                attempts += 1
            
            # Si le batch est complet, le sauvegarder
            if len(batch) == self.batch_size:
                batches.append(batch)
        
        # Retourner les batches
        for batch in batches:
            yield batch
    
    def __len__(self):
        return self.num_batches


# =========================================================
# EVALUATION METRICS: BLEU-4 + BERTScore
# =========================================================
def simple_tokenize(s: str):
    """Tokenisation simple et robuste pour BLEU."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t]


def compute_bleu4(pred_texts, ref_texts):
    """
    Calcule BLEU-4 moyen pour une liste de prédictions vs références.
    """
    if sentence_bleu is None:
        return 0.0
    
    smoothing_function = SmoothingFunction().method1
    scores = []
    
    for pred, ref in zip(pred_texts, ref_texts):
        pred_tok = simple_tokenize(pred)
        ref_tok = simple_tokenize(ref)
        
        # Skip si tokens vides
        if len(pred_tok) == 0 or len(ref_tok) == 0:
            scores.append(0.0)
            continue
        
        bleu = sentence_bleu([ref_tok], pred_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
        scores.append(bleu)
    
    return float(np.mean(scores)) if scores else 0.0


def compute_bertscore(reference_list, hypothesis_list, model_type="bert-base-multilingual-cased"):
    """
    Calcule BERTScore moyen entre références et hypothèses.
    """
    if bert_score is None:
        return 0.0, 0.0, 0.0
    
    try:
        P, R, F1 = bert_score(hypothesis_list, reference_list, lang="en", model_type=model_type, verbose=False)
        return float(P.mean()), float(R.mean()), float(F1.mean())
    except:
        return 0.0, 0.0, 0.0


@torch.no_grad()
def eval_bleu_retrieval(val_graphs_path, val_emb_dict, val_text_dict, 
                        train_emb_dict, train_text_dict, model, device, batch_size=64):
    """
    Évalue BLEU-4 en faisant un retrieval top-1 des captions d'entraînement
    pour chaque molécule de validation.
    
    Args:
        val_graphs_path: chemin vers les graphes de validation
        val_emb_dict: id -> validation text embedding
        val_text_dict: id -> validation text (description)
        train_emb_dict: id -> train text embedding
        train_text_dict: id -> train text (description)
        model: le modèle Graph Transformer
        device: cuda/cpu
        batch_size: batch size pour l'inférence
    
    Returns:
        dict avec "BLEU4" et "BLEU4_avg"
    """
    model.eval()
    
    # Charger le dataset de validation
    ds = PreprocessedGraphDataset(val_graphs_path, val_emb_dict)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Construire la matrice d'embeddings train (alignée avec train_ids)
    train_ids = sorted(train_emb_dict.keys())
    train_txt_emb = torch.from_numpy(np.array([train_emb_dict[id_] for id_ in train_ids], dtype=np.float32)).to(device)
    train_txt_emb = F.normalize(train_txt_emb, dim=-1)
    
    pred_texts, ref_texts = [], []
    offset = 0
    
    for graphs, _text_emb in dl:
        graphs = graphs.to(device)
        
        # Encode les graphes
        mol_emb = model(graphs)  # [B, D]
        mol_emb = F.normalize(mol_emb, dim=-1)
        
        # Retrieval top-1: quelle train caption est plus similaire?
        sims = mol_emb @ train_txt_emb.t()  # [B, Ntrain]
        best_indices = sims.argmax(dim=1).tolist()
        
        # Récupérer les IDs de validation du batch
        # Important: on suppose que ds.ids existe et est aligné avec le dataset
        if hasattr(ds, 'ids'):
            batch_val_ids = ds.ids[offset: offset + graphs.num_graphs]
        else:
            # Fallback: on ne peut pas évaluer BLEU sans les IDs
            print("⚠️  PreprocessedGraphDataset n'a pas d'attribut 'ids', BLEU non disponible")
            return {"BLEU4": 0.0}
        
        offset += graphs.num_graphs
        
        # Construire les paires pred/ref
        for j, val_id in enumerate(batch_val_ids):
            train_id = train_ids[best_indices[j]]
            pred_text = train_text_dict.get(str(train_id), "")
            ref_text = val_text_dict.get(str(val_id), "")
            
            if pred_text and ref_text:
                pred_texts.append(pred_text)
                ref_texts.append(ref_text)
    
    # Calculer BLEU-4
    bleu4 = compute_bleu4(pred_texts, ref_texts) if pred_texts else 0.0
    
    return {"BLEU4": bleu4}


def get_temperature_schedule(epoch, total_epochs, init_temp=0.1, final_temp=0.07):
    """
    Retourne la température pour l'epoch courant.
    Décroissance linéaire de init_temp vers final_temp.
    """
    if not args.temp_schedule:
        return args.temp
    
    # Décroissance linéaire
    progress = epoch / max(total_epochs - 1, 1)
    temp = init_temp - (init_temp - final_temp) * progress
    return max(temp, final_temp)


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
# GRAPH AUGMENTATION
# =========================================================
class GraphAugmentation(nn.Module):
    """
    Augmentation légère de graphes pour contrastive learning multi-vues.
    """
    def __init__(self, edge_dropout=0.1, node_dropout=0.05):
        super().__init__()
        self.edge_dropout = edge_dropout
        self.node_dropout = node_dropout
    
    def forward(self, batch, training=True):
        """
        Applique des augmentations aléatoires au batch de graphes.
        """
        if not training:
            return batch
        
        # Copie le batch pour ne pas le modifier en place
        batch_aug = batch.clone()
        
        # Edge dropout: supprime aléatoirement des edges
        if self.edge_dropout > 0:
            mask = torch.rand(batch_aug.edge_index.size(1)) > self.edge_dropout
            batch_aug.edge_index = batch_aug.edge_index[:, mask]
            if batch_aug.edge_attr is not None:
                batch_aug.edge_attr = batch_aug.edge_attr[mask]
        
        # Node feature dropout: masque aléatoirement des features
        if self.node_dropout > 0 and batch_aug.x is not None:
            mask = torch.rand(batch_aug.x.size()) > self.node_dropout
            batch_aug.x = batch_aug.x * mask.float()
        
        return batch_aug


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
        
        # Softmax groupé par graphe (stable numériquement via PyG softmax)
        attn_weights = softmax(attn_logits, batch, dim=0)  # [num_nodes, 1]
        
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
def eval_retrieval(data_path, emb_dict, model, device, temp=0.1, compute_loss=True):
    """
    Évalue le modèle sur le validation set.
    
    Args:
        data_path: Chemin vers les graphes de validation
        emb_dict: Dictionnaire des embeddings de texte
        model: Modèle à évaluer
        device: Device (cuda/cpu)
        temp: Température pour la loss contrastive (si compute_loss=True)
        compute_loss: Si True, calcule aussi la validation loss
    
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
        
        # Calcul de la validation loss si demandé
        if compute_loss:
            val_loss = contrastive_loss(mol_vec, text_emb, temperature=temp)
            total_val_loss += val_loss.item() * graphs.num_graphs
            total_count += graphs.num_graphs
        
        all_mol.append(mol_vec)
        all_txt.append(F.normalize(text_emb, dim=-1))
        
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    # Métriques de retrieval
    sims = all_txt @ all_mol.t()
    
    ranks = sims.argsort(dim=-1, descending=True)
    N = all_txt.size(0)
    correct_indices = torch.arange(N, device=device).unsqueeze(1)
    
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
    print(f"Loss: Contrastive (Temp={args.temp})")
    print(f"Architecture: Improved Transformer (hidden={args.hidden}, layers={args.layers}, heads={args.heads})")
    print(f"Hard Negative Mining: {'Enabled' if args.hard_negatives else 'Disabled'}")
    if args.hard_negatives:
        print(f"  - Hard Ratio: {args.hard_ratio*100:.0f}%")
        print(f"  - Hardness K: {args.hardness_k}")
        print(f"  - Curriculum Start: Epoch {args.curriculum_epoch if args.curriculum_epoch > 0 else 'from start'}")
    print(f"Temperature Schedule: {'Enabled (0.1 → 0.07)' if args.temp_schedule else 'Disabled'}")
    print(f"Graph Augmentation: {'Enabled' if args.use_augmentation else 'Disabled'}")
    print(f"BLEU-4 + BERTScore Eval: {'Enabled' if args.eval_bleu_bert else 'Disabled'}")
    print(f"Rerank Top-K: {args.rerank_topk}")
    
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    
    # Charger les descriptions textuelles (nécessaires pour BLEU-4)
    train_text_dict = load_id2text(TRAIN_GRAPHS) if args.eval_bleu_bert else {}
    val_text_dict = load_id2text(VAL_GRAPHS) if args.eval_bleu_bert and os.path.exists(VAL_GRAPHS) else {}
    
    if not os.path.exists(TRAIN_GRAPHS):
        raise FileNotFoundError(f"Graphs not found at {TRAIN_GRAPHS}")

    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    
    # Créer le DataLoader avec ou sans Hard Negative Sampler
    if args.hard_negatives and args.curriculum_epoch == 0:
        # Hard Negative Mining dès le début
        print("\n🎯 Initialisation du Hard Negative Sampler...")
        sampler = HardNegativeSampler(
            train_emb, 
            batch_size=args.batch_size,
            hard_ratio=args.hard_ratio,
            hardness_k=args.hardness_k,
            dataset=train_ds  # CRITICAL FIX: passer le dataset
        )
        train_dl = DataLoader(
            train_ds,
            batch_sampler=sampler,
            collate_fn=collate_fn
        )
        print(f"✅ DataLoader configuré avec Hard Negative Mining\n")
    else:
        # Mode standard (shuffle random) ou curriculum learning
        if args.hard_negatives and args.curriculum_epoch > 0:
            print(f"\n📚 Curriculum Learning: Random sampling jusqu'à l'epoch {args.curriculum_epoch}")
            print(f"   Hard Negative Mining démarrera à l'epoch {args.curriculum_epoch + 1}\n")
        train_dl = DataLoader(
            train_ds, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn, 
            drop_last=True
        )
    
    # Sauvegarder le sampler pour curriculum learning
    hard_sampler = None
    if args.hard_negatives and args.curriculum_epoch > 0:
        hard_sampler = None  # Sera créé plus tard

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
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    # Gestion de la reprise d'entraînement
    start_epoch = 0
    best_mrr = 0.0
    training_logs = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "temp": args.temp,
            "hidden": args.hidden,
            "layers": args.layers,
            "heads": args.heads
        },
        "epochs": []
    }
    
    # Charger le checkpoint si spécifié
    if args.resume_from:
        checkpoint_path = f"{base_path}/{args.resume_from}" if not args.resume_from.startswith('/') else args.resume_from
        
        # Essayer d'abord de charger le checkpoint complet
        full_checkpoint_path = f"{OUTPUT_DIR}/checkpoint.pt"
        if os.path.exists(full_checkpoint_path):
            print(f"\n🔄 Chargement du checkpoint complet: {full_checkpoint_path}")
            checkpoint = torch.load(full_checkpoint_path, map_location=DEVICE)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restaurer le scheduler si possible
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except:
                    print("⚠️  Impossible de restaurer l'état du scheduler, utilisation de l'état actuel")
            
            best_mrr = checkpoint.get('best_mrr', 0.0)
            start_epoch = checkpoint.get('epoch', 0)
            restored_lr = checkpoint.get('learning_rate', args.lr)
            
            # Restaurer le LR dans l'optimizer (important car le scheduler peut ne pas être restauré)
            for param_group in optimizer.param_groups:
                param_group['lr'] = restored_lr
            
            print("✅ Checkpoint complet chargé avec succès")
            print(f"📊 Epoch: {start_epoch}")
            print(f"📊 Meilleur MRR: {best_mrr:.4f}")
            print(f"📊 Learning Rate restauré: {restored_lr:.6f}")
            
            # Charger les logs pour récupérer l'historique complet
            logs_path = f"{OUTPUT_DIR}/training_logs.json"
            if os.path.exists(logs_path):
                with open(logs_path, 'r') as f:
                    old_logs = json.load(f)
                    if old_logs.get("epochs"):
                        training_logs["epochs"] = old_logs["epochs"]
                        print(f"📊 Historique chargé: {len(old_logs['epochs'])} epochs")
            
            # Utiliser start_epoch manuel si fourni
            if args.start_epoch > 0:
                start_epoch = args.start_epoch - 1
                print(f"📊 Utilisation de l'epoch de départ manuel: {start_epoch + 1}")
                
        elif os.path.exists(checkpoint_path):
            # Fallback: charger seulement le modèle (ancien format)
            print(f"\n🔄 Chargement du modèle uniquement: {checkpoint_path}")
            print("⚠️  Checkpoint complet non trouvé, chargement du modèle seulement")
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print("✅ Modèle chargé avec succès")
            
            # Essayer de charger les logs pour récupérer l'historique et le LR
            logs_path = f"{OUTPUT_DIR}/training_logs.json"
            if os.path.exists(logs_path):
                with open(logs_path, 'r') as f:
                    old_logs = json.load(f)
                    if old_logs.get("epochs"):
                        start_epoch = len(old_logs["epochs"])
                        best_mrr = old_logs.get("best_mrr", 0.0)
                        training_logs["epochs"] = old_logs["epochs"]
                        
                        # Récupérer le dernier LR depuis les logs
                        if old_logs["epochs"]:
                            last_epoch_log = old_logs["epochs"][-1]
                            last_lr = last_epoch_log.get("learning_rate", args.lr)
                            # Ajuster le LR de l'optimizer
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = last_lr
                            print(f"📊 Learning Rate restauré depuis les logs: {last_lr:.6f}")
                        
                        print(f"📊 Reprise depuis l'epoch {start_epoch + 1}")
                        print(f"📊 Meilleur MRR précédent: {best_mrr:.4f}")
            
            # Utiliser start_epoch si fourni manuellement
            if args.start_epoch > 0:
                start_epoch = args.start_epoch - 1
                print(f"📊 Utilisation de l'epoch de départ manuel: {start_epoch + 1}")
        else:
            print(f"⚠️  Checkpoint non trouvé: {checkpoint_path}")
            print("   Démarrage d'un nouvel entraînement...")
    
    # Ajuster le nombre total d'epochs si on reprend
    total_epochs = args.epochs
    if start_epoch > 0:
        print(f"\n📈 Entraînement: epochs {start_epoch + 1} à {total_epochs}")
    
    for ep in range(start_epoch, total_epochs):
        # Curriculum Learning: Passer au Hard Negative Mining à l'epoch spécifiée
        if args.hard_negatives and args.curriculum_epoch > 0 and ep == args.curriculum_epoch:
            print(f"\n🎓 CURRICULUM SWITCH: Activation du Hard Negative Mining à l'epoch {ep + 1}")
            print("🎯 Création du Hard Negative Sampler...\n")
            hard_sampler = HardNegativeSampler(
                train_emb,
                batch_size=args.batch_size,
                hard_ratio=args.hard_ratio,
                hardness_k=args.hardness_k,
                dataset=train_ds  # CRITICAL FIX: passer le dataset
            )
            train_dl = DataLoader(
                train_ds,
                batch_sampler=hard_sampler,
                collate_fn=collate_fn
            )
            print(f"✅ DataLoader mis à jour avec Hard Negative Mining\n")
        
        # Température adaptative
        current_temp = get_temperature_schedule(ep, args.epochs)
        
        train_loss = train_epoch(model, train_dl, optimizer, DEVICE, current_temp)
        
        # Récupérer le learning rate actuel
        current_lr = optimizer.param_groups[0]['lr']
        
        val_metrics = {}
        if os.path.exists(VAL_GRAPHS) and os.path.exists(VAL_EMB_CSV):
            val_emb = load_id2emb(VAL_EMB_CSV)
            # Calculer les métriques de retrieval ET la validation loss
            val_metrics = eval_retrieval(VAL_GRAPHS, val_emb, model, DEVICE, temp=args.temp, compute_loss=True)
            
            # Évaluer BLEU-4 si activé
            if args.eval_bleu_bert and train_text_dict and val_text_dict:
                bleu_results = eval_bleu_retrieval(
                    VAL_GRAPHS, val_emb, val_text_dict,
                    train_emb, train_text_dict,
                    model, DEVICE, batch_size=args.batch_size
                )
                val_metrics.update(bleu_results)
            
            # Scheduler step basé sur MRR
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_metrics["MRR"])
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"  [LR Reduced] {old_lr:.2e} -> {new_lr:.2e}")
            
            if val_metrics["MRR"] > best_mrr:
                best_mrr = val_metrics["MRR"]
                # Sauvegarder le modèle (pour compatibilité)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                # Sauvegarder un checkpoint complet (modèle + optimizer + état)
                checkpoint = {
                    'epoch': ep + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mrr': best_mrr,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'val_mrr': val_metrics["MRR"]  # Pour restaurer l'état du scheduler
                }
                # Essayer de sauvegarder le scheduler si possible
                try:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                except:
                    pass  # ReduceLROnPlateau n'a pas toujours de state_dict
                torch.save(checkpoint, CHECKPOINT_SAVE_PATH)
                print(f"  [New Best Model Saved] MRR: {best_mrr:.4f}")

        # Enregistrer les logs
        epoch_log = {
            "epoch": ep + 1,
            "train_loss": float(train_loss),
            "learning_rate": float(current_lr),
            "val_loss": float(val_metrics.get("val_loss", 0.0)),
            "val_mrr": float(val_metrics.get("MRR", 0.0)),
            "val_r1": float(val_metrics.get("R@1", 0.0)),
            "val_r5": float(val_metrics.get("R@5", 0.0)),
            "val_r10": float(val_metrics.get("R@10", 0.0)),
            "val_bleu4": float(val_metrics.get("BLEU4", 0.0))
        }
        training_logs["epochs"].append(epoch_log)
        
        # Sauvegarder les logs à chaque epoch (pour pouvoir les visualiser en temps réel)
        with open(LOGS_SAVE_PATH, 'w') as f:
            json.dump(training_logs, f, indent=2)
        
        val_str = f"Loss: {val_metrics.get('val_loss', 0.0):.4f}, " if val_metrics.get('val_loss') else ""
        val_str += f"MRR: {val_metrics.get('MRR', 0.0):.4f}, R@1: {val_metrics.get('R@1', 0.0):.4f}"
        if args.eval_bleu_bert and val_metrics.get("BLEU4"):
            val_str += f", BLEU-4: {val_metrics.get('BLEU4', 0.0):.4f}"
        print(f"Epoch {ep+1:02d}/{total_epochs} | Train Loss: {train_loss:.4f} | Val: {val_str}")

    training_logs["best_mrr"] = float(best_mrr)
    with open(LOGS_SAVE_PATH, 'w') as f:
        json.dump(training_logs, f, indent=2)
    
    if start_epoch > 0:
        print(f"\n✅ Entraînement repris et complété. Epochs {start_epoch + 1} à {total_epochs} terminés.")
    else:
        print(f"\nTraining Complete. Best Model saved to {MODEL_SAVE_PATH}")
    print(f"Training logs saved to {LOGS_SAVE_PATH}")
    print(f"Best MRR: {best_mrr:.4f}")

if __name__ == "__main__":
    main()