# train_retrieve_top20_T5.py
# ---------------------------------------------------------
# Pipeline: (1) use a pretrained graph->embedding encoder (your contrastive model)
#           (2) retrieve top-K training captions (K=20 by default)
#           (3) train a T5 (rewrite model) to generate the GT caption from retrieved candidates
#
# You can start with:
# !python train_retrieve_top20_T5.py --env colab \
#   --graph_ckpt /content/drive/MyDrive/data/GT_Contrast/contrastive_model.pt \
#   --epochs 5 --batch_size 8 --topk 20 --t5_name t5-small
#
# Notes:
# - This script assumes train_graphs.pkl and validation_graphs.pkl contain graph objects
#   with attributes: id, description, x, edge_index, edge_attr (PyG Data-like).
# - It also assumes train_embeddings.csv / validation_embeddings.csv exist and map id->text embedding.
# - We freeze the graph encoder and train only T5.
# ---------------------------------------------------------

import argparse
import os
import json
import pickle
import re
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import TransformerConv, LayerNorm
from torch_geometric.utils import scatter, softmax

# Text metrics (optional)
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download("punkt", quiet=True)
except Exception:
    sentence_bleu = None

# Transformers (T5)
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)

from data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn


# =========================================================
# Utils: text / BLEU
# =========================================================
def simple_tokenize(s: str):
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t]


def compute_bleu4(pred_texts: List[str], ref_texts: List[str]) -> float:
    if sentence_bleu is None:
        return 0.0
    smoothing = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(pred_texts, ref_texts):
        pt = simple_tokenize(pred)
        rt = simple_tokenize(ref)
        if not pt or not rt:
            scores.append(0.0)
            continue
        scores.append(
            sentence_bleu([rt], pt, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        )
    return float(np.mean(scores)) if scores else 0.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Load graphs + descriptions from PKL
# =========================================================
def load_graphs_pkl(path: str):
    with open(path, "rb") as f:
        graphs = pickle.load(f)
    return graphs


def load_id2text_from_pkl(path: str, id_attr="id", text_attr="description") -> Dict[str, str]:
    graphs = load_graphs_pkl(path)
    d = {}
    missing = 0
    for g in graphs:
        gid = getattr(g, id_attr, None)
        if gid is None:
            continue
        gid = str(gid)
        desc = getattr(g, text_attr, "") or ""
        if not desc:
            missing += 1
        d[gid] = str(desc)
    if missing > 0:
        print(f"⚠️  {missing}/{len(graphs)} graphs missing descriptions in {path}")
    print(f"✅ Loaded {len(d)} descriptions from {path}")
    return d


def load_ids_in_pkl_order(path: str, id_attr="id") -> List[str]:
    graphs = load_graphs_pkl(path)
    ids = []
    for g in graphs:
        gid = getattr(g, id_attr, None)
        ids.append(str(gid) if gid is not None else "")
    return ids


# =========================================================
# Model: same graph encoder as your contrastive setup
# =========================================================
class ImprovedAtomEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(120, emb_dim),  # atomic_num
            nn.Embedding(10, emb_dim),   # chirality
            nn.Embedding(12, emb_dim),   # degree
            nn.Embedding(12, emb_dim),   # formal charge
            nn.Embedding(10, emb_dim),   # num_hs
            nn.Embedding(6, emb_dim),    # radical
            nn.Embedding(10, emb_dim),   # hybridization
            nn.Embedding(2, emb_dim),    # is_aromatic
            nn.Embedding(2, emb_dim)     # is_in_ring
        ])
        self.feature_weights = nn.Parameter(torch.ones(len(self.embeddings)))
        self.combine = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU()
        )

    def forward(self, x):
        embs = []
        for i, emb in enumerate(self.embeddings):
            embs.append(emb(x[:, i]) * torch.sigmoid(self.feature_weights[i]))
        out = sum(embs)
        return self.combine(out)


class ImprovedBondEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(20, emb_dim),  # bond_type
            nn.Embedding(10, emb_dim),  # stereo
            nn.Embedding(2, emb_dim)    # is_conjugated
        ])
        self.feature_weights = nn.Parameter(torch.ones(len(self.embeddings)))
        self.combine = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU()
        )

    def forward(self, edge_attr):
        embs = []
        for i, emb in enumerate(self.embeddings):
            embs.append(emb(edge_attr[:, i]) * torch.sigmoid(self.feature_weights[i]))
        out = sum(embs)
        return self.combine(out)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, batch):
        logits = self.attention(x)
        w = softmax(logits, batch, dim=0)
        return scatter(x * w, batch, dim=0, reduce="sum")


class ImprovedMolGraphTransformer(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.atom_encoder = ImprovedAtomEncoder(hidden)
        self.bond_encoder = ImprovedBondEncoder(hidden)
        self.pos_encoder = nn.Embedding(12, hidden)

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.drops = nn.ModuleList()

        for _ in range(layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden,
                    out_channels=hidden // heads,
                    heads=heads,
                    edge_dim=hidden,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.lns.append(LayerNorm(hidden))
            self.drops.append(nn.Dropout(dropout))

        self.pool = AttentionPooling(hidden)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, batch):
        x = self.atom_encoder(batch.x)
        degree = batch.x[:, 2].long().clamp(0, 11)
        x = x + self.pos_encoder(degree)
        edge_attr = self.bond_encoder(batch.edge_attr)

        for conv, ln, drop in zip(self.convs, self.lns, self.drops):
            residual = x
            x = conv(x, batch.edge_index, edge_attr)
            x = ln(x)
            x = F.relu(x)
            x = drop(x)
            if x.shape == residual.shape:
                x = x + residual

        g = self.pool(x, batch.batch)
        g = self.proj(g)
        return F.normalize(g, dim=-1)


# =========================================================
# Build retrieval candidates
# =========================================================
@torch.no_grad()
def compute_mol_embeddings(
    model: nn.Module,
    graphs_pkl_path: str,
    emb_dict: Dict[str, np.ndarray],
    device: str,
    batch_size: int = 128,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Uses PreprocessedGraphDataset to load graphs (aligned to emb_dict).
    Returns mol_emb matrix [N, D] and ids list aligned with that order.
    """
    ds = PreprocessedGraphDataset(graphs_pkl_path, emb_dict)

    # Ensure we have ids aligned in pkl order; if ds has ids/graph_ids use them.
    # Otherwise fallback to reading ids in pickle order.
    if hasattr(ds, "ids") and isinstance(ds.ids, (list, tuple)) and len(ds.ids) == len(ds):
        ids = [str(x) for x in ds.ids]
    elif hasattr(ds, "graph_ids") and isinstance(ds.graph_ids, (list, tuple)) and len(ds.graph_ids) == len(ds):
        ids = [str(x) for x in ds.graph_ids]
    else:
        ids = load_ids_in_pkl_order(graphs_pkl_path)

        # If lengths mismatch, still proceed but warn
        if len(ids) != len(ds):
            print("⚠️  Could not align ds order with PKL ids perfectly. BLEU evaluation may be off.")
            ids = ids[:len(ds)]

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_embs = []
    for graphs, _txt in dl:
        graphs = graphs.to(device)
        e = model(graphs)
        all_embs.append(e.detach().cpu())
    mol_emb = torch.cat(all_embs, dim=0)
    return mol_emb, ids


def build_train_text_matrix(train_emb: Dict[str, np.ndarray], device: str):
    train_ids = sorted(train_emb.keys())
    mat = torch.from_numpy(np.array([train_emb[i] for i in train_ids], dtype=np.float32)).to(device)
    mat = F.normalize(mat, dim=-1)
    return train_ids, mat


@torch.no_grad()
def retrieve_topk_indices(
    mol_emb: torch.Tensor,
    train_txt_emb: torch.Tensor,
    topk: int,
    device: str,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    mol_emb: [N, D] on CPU
    train_txt_emb: [M, D] on GPU (or device)
    returns indices [N, topk] (CPU) into train_ids
    """
    mol_emb = F.normalize(mol_emb, dim=-1)
    N = mol_emb.size(0)
    topk = min(topk, train_txt_emb.size(0))
    out = []

    for i in range(0, N, chunk_size):
        mb = mol_emb[i:i + chunk_size].to(device)
        sims = mb @ train_txt_emb.t()
        idx = sims.topk(topk, dim=1).indices.detach().cpu()
        out.append(idx)

    return torch.cat(out, dim=0)


def make_rewrite_input(candidates: List[str], max_cands: int = 20) -> str:
    """
    Format input for T5.
    Keep it simple and stable.
    """
    cands = candidates[:max_cands]
    parts = []
    for i, c in enumerate(cands, 1):
        c = (c or "").strip()
        if not c:
            continue
        parts.append(f"[{i}] {c}")
    joined = " ".join(parts)
    return f"rewrite: {joined}"


# =========================================================
# Dataset for T5 rewrite
# =========================================================
class T5RewriteDataset(Dataset):
    def __init__(
        self,
        split_ids: List[str],
        split_targets: Dict[str, str],
        topk_train_ids: List[str],
        topk_indices: torch.Tensor,   # [N, K] indices into train_ids list
        train_text_dict: Dict[str, str],
        tokenizer,
        max_input_len: int = 512,
        max_target_len: int = 128,
    ):
        self.ids = split_ids
        self.targets = split_targets
        self.topk_train_ids = topk_train_ids
        self.topk_indices = topk_indices
        self.train_text_dict = train_text_dict
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

        assert len(self.ids) == self.topk_indices.size(0), "IDs and topk_indices must align"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        gid = str(self.ids[idx])
        tgt = (self.targets.get(gid, "") or "").strip()

        inds = self.topk_indices[idx].tolist()
        cand_ids = [self.topk_train_ids[j] for j in inds]
        candidates = [self.train_text_dict.get(str(cid), "") for cid in cand_ids]
        inp = make_rewrite_input(candidates, max_cands=len(candidates))

        model_in = self.tokenizer(
            inp,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt,
                max_length=self.max_target_len,
                truncation=True,
                padding=False,
                return_tensors=None,
            )

        return {
            "input_ids": model_in["input_ids"],
            "attention_mask": model_in["attention_mask"],
            "labels": labels["input_ids"],
            "gid": gid,
            "target_text": tgt,
        }


@dataclass
class T5Collator:
    tokenizer: object
    label_pad_token_id: int = -100

    def __call__(self, batch):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        enc = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        lab = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )["input_ids"]

        # Replace padding token id by -100 for loss masking
        lab = lab.masked_fill(lab == self.tokenizer.pad_token_id, self.label_pad_token_id)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": lab,
            "gids": [b["gid"] for b in batch],
            "target_texts": [b["target_text"] for b in batch],
        }


# =========================================================
# Train / eval loops
# =========================================================
def run_eval_bleu(model, tokenizer, dl, device, max_gen_len=128, num_beams=2, eval_max_batches=0):
    model.eval()
    preds, refs = [], []
    n_batches = 0

    with torch.no_grad():
        for batch in dl:
            n_batches += 1
            if eval_max_batches > 0 and n_batches > eval_max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_gen_len,
                num_beams=num_beams,
            )
            pred_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            preds.extend(pred_texts)
            refs.extend(batch["target_texts"])

    bleu = compute_bleu4(preds, refs)
    return bleu


def train_one_epoch(model, dl, optimizer, scheduler, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        bs = input_ids.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs

    return total_loss / max(total_count, 1)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Retrieve topK + T5 rewrite training")
    parser.add_argument("--env", type=str, default="local", choices=["local", "colab"])
    parser.add_argument("--seed", type=int, default=42)

    # Paths
    parser.add_argument("--graph_ckpt", type=str, required=True, help="Path to pretrained contrastive graph encoder .pt (state_dict)")
    parser.add_argument("--output_dir", type=str, default=None)

    # Graph encoder arch (must match ckpt)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)

    # Retrieval
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--retrieval_bs", type=int, default=128)

    # T5
    parser.add_argument("--t5_name", type=str, default="t5-small")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Eval
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_max_batches", type=int, default=0, help="If >0, evaluate only first N batches")

    args = parser.parse_args()
    set_seed(args.seed)

    base_path = "/content/drive/MyDrive/data" if args.env == "colab" else "data"
    TRAIN_GRAPHS = f"{base_path}/train_graphs.pkl"
    VAL_GRAPHS = f"{base_path}/validation_graphs.pkl"
    TRAIN_EMB_CSV = f"{base_path}/train_embeddings.csv"
    VAL_EMB_CSV = f"{base_path}/validation_embeddings.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.output_dir is None:
        args.output_dir = f"{base_path}/retrieveT5_top{args.topk}"
    os.makedirs(args.output_dir, exist_ok=True)

    LOG_PATH = os.path.join(args.output_dir, "training_logs.json")
    BEST_PATH = os.path.join(args.output_dir, "best_t5.pt")

    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"Graph ckpt: {args.graph_ckpt}")
    print(f"Retrieval topk: {args.topk}")
    print(f"T5: {args.t5_name}")

    # --- Load embeddings dicts
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else {}

    # --- Load text dicts (targets + candidate captions)
    train_text = load_id2text_from_pkl(TRAIN_GRAPHS)
    val_text = load_id2text_from_pkl(VAL_GRAPHS) if os.path.exists(VAL_GRAPHS) else {}

    # --- Build graph encoder & load ckpt
    emb_dim = len(next(iter(train_emb.values())))
    graph_encoder = ImprovedMolGraphTransformer(
        hidden=args.hidden,
        out_dim=emb_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=0.1,
    ).to(device)

    sd = torch.load(args.graph_ckpt, map_location=device)
    graph_encoder.load_state_dict(sd)
    graph_encoder.eval()
    for p in graph_encoder.parameters():
        p.requires_grad = False

    # --- Prepare train text embedding matrix for retrieval
    train_ids_sorted, train_txt_mat = build_train_text_matrix(train_emb, device=device)

    # --- Compute mol embeddings for train & val
    print("\n🔎 Computing train mol embeddings...")
    train_mol_emb, train_ids_in_order = compute_mol_embeddings(
        graph_encoder, TRAIN_GRAPHS, train_emb, device=device, batch_size=args.retrieval_bs
    )
    print(f"Train mol emb: {train_mol_emb.shape}")

    print("\n🔎 Computing val mol embeddings...")
    val_mol_emb, val_ids_in_order = compute_mol_embeddings(
        graph_encoder, VAL_GRAPHS, val_emb, device=device, batch_size=args.retrieval_bs
    )
    print(f"Val mol emb: {val_mol_emb.shape}")

    # --- Retrieve topK indices (val/train molecules -> topK train captions)
    print("\n🧲 Retrieving top-K candidates for TRAIN...")
    train_topk_idx = retrieve_topk_indices(
        train_mol_emb, train_txt_mat, topk=args.topk, device=device, chunk_size=1024
    )
    print("✅ Train retrieval done.")

    print("\n🧲 Retrieving top-K candidates for VAL...")
    val_topk_idx = retrieve_topk_indices(
        val_mol_emb, train_txt_mat, topk=args.topk, device=device, chunk_size=1024
    )
    print("✅ Val retrieval done.")

    # --- Build tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(args.t5_name)
    t5 = AutoModelForSeq2SeqLM.from_pretrained(args.t5_name).to(device)

    # --- Build datasets
    # Targets: use description from pkl dict keyed by id
    train_ds = T5RewriteDataset(
        split_ids=train_ids_in_order,
        split_targets=train_text,
        topk_train_ids=train_ids_sorted,
        topk_indices=train_topk_idx,
        train_text_dict=train_text,
        tokenizer=tokenizer,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
    )
    val_ds = T5RewriteDataset(
        split_ids=val_ids_in_order,
        split_targets=val_text,
        topk_train_ids=train_ids_sorted,
        topk_indices=val_topk_idx,
        train_text_dict=train_text,
        tokenizer=tokenizer,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
    )

    collator = T5Collator(tokenizer=tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=max(2, args.batch_size), shuffle=False, collate_fn=collator)

    # --- Optim / scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in t5.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in t5.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    total_steps = args.epochs * len(train_dl)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Train
    logs = {
        "config": vars(args),
        "epochs": []
    }
    best_bleu = -1.0

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(t5, train_dl, optimizer, scheduler, device=device, grad_clip=args.grad_clip)

        row = {"epoch": ep, "train_loss": float(tr_loss)}
        print(f"\nEpoch {ep}/{args.epochs} | train_loss={tr_loss:.4f}")

        if (ep % args.eval_every) == 0 and len(val_ds) > 0:
            bleu = run_eval_bleu(
                t5, tokenizer, val_dl, device=device,
                max_gen_len=args.max_target_len,
                num_beams=args.num_beams,
                eval_max_batches=args.eval_max_batches
            )
            row["val_bleu4"] = float(bleu)
            print(f"  ✅ val_BLEU4={bleu:.4f}")

            if bleu > best_bleu:
                best_bleu = bleu
                torch.save(t5.state_dict(), BEST_PATH)
                print(f"  💾 New best saved: {BEST_PATH}")

        logs["epochs"].append(row)
        with open(LOG_PATH, "w") as f:
            json.dump(logs, f, indent=2)

    print("\nDone.")
    print(f"Best BLEU4: {best_bleu:.4f}")
    print(f"Logs: {LOG_PATH}")
    print(f"Best model: {BEST_PATH}")


if __name__ == "__main__":
    main()
