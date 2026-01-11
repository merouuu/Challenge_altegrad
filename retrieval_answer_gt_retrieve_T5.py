# generate_submission_retrieveTopK_T5.py
# ---------------------------------------------------------
# Inference / submission script for the NEW model:
#   Graph Encoder (ImprovedMolGraphTransformer) + Retrieve Top-K + T5 rewriter
#
# Steps:
# 1) Encode TEST graphs -> mol embeddings
# 2) Retrieve Top-K TRAIN captions using cosine similarity with TRAIN text embeddings (train_embeddings.csv)
# 3) Build T5 input: "rewrite: [1] cand1 [2] cand2 ... [K] candK"
# 4) Use T5 to generate final caption
# 5) Save Kaggle submission CSV: columns ['ID', 'description']
#
# Example (Colab):
# !python generate_submission_retrieveTopK_T5.py --env colab \
#   --graph_ckpt /content/drive/MyDrive/data/GT_Contrast/contrastive_model.pt \
#   --t5_name t5-small \
#   --t5_ckpt /content/drive/MyDrive/data/retrieveT5_top20/best_t5.pt \
#   --topk 20 --batch_size 32 --num_beams 2
# ---------------------------------------------------------

import argparse
import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from data_utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn
)

# Import the graph encoder architecture (must match training)
from train_gt_contrast import ImprovedMolGraphTransformer


# =========================================================
# Text formatting (MUST match training)
# =========================================================
def make_rewrite_input(candidates, max_cands=20):
    """
    Format input for T5 rewrite model.
    Keep it identical to training to avoid distribution shift.
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
# Utilities
# =========================================================
def _to_tensor_stack(list_of_vecs, device):
    """
    load_id2emb sometimes returns numpy arrays, sometimes torch tensors.
    This makes a normalized torch matrix on device.
    """
    if len(list_of_vecs) == 0:
        return torch.empty(0, device=device)

    v0 = list_of_vecs[0]
    if isinstance(v0, torch.Tensor):
        mat = torch.stack(list_of_vecs, dim=0).to(device)
    else:
        mat = torch.from_numpy(np.array(list_of_vecs, dtype=np.float32)).to(device)
    return mat


@torch.no_grad()
def encode_molecules(model, dataset, device, batch_size=64):
    """
    dataset is PreprocessedGraphDataset(test_graphs.pkl) (no text_emb)
    We rely on dataset.ids to keep order stable.
    """
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_embs = []
    for batch in dl:
        # Depending on collate_fn, test loader may return:
        # - graphs only
        # - (graphs, text_emb) for train/val
        graphs = batch[0] if isinstance(batch, (tuple, list)) else batch
        graphs = graphs.to(device)
        emb = model(graphs)  # already normalized in your forward()
        all_embs.append(emb.detach())

    mol_embs = torch.cat(all_embs, dim=0)
    return mol_embs


def batched_topk_retrieval(mol_embs, train_txt_embs, topk=20, chunk_size=1024):
    """
    mol_embs: [N, D] on device
    train_txt_embs: [M, D] on device
    returns indices: [N, topk] on CPU
    """
    mol_embs = F.normalize(mol_embs, dim=-1)
    train_txt_embs = F.normalize(train_txt_embs, dim=-1)

    N = mol_embs.size(0)
    topk = min(topk, train_txt_embs.size(0))
    out = []

    for i in range(0, N, chunk_size):
        mb = mol_embs[i:i + chunk_size]
        sims = mb @ train_txt_embs.t()  # [B, M]
        idx = sims.topk(topk, dim=1).indices.detach().cpu()
        out.append(idx)

    return torch.cat(out, dim=0)


# =========================================================
# Dataset for T5 generation inputs (test-time)
# =========================================================
class TestRewriteDataset(Dataset):
    def __init__(self, test_ids, topk_indices, train_ids, train_id2desc, tokenizer, max_input_len=512, topk=20):
        self.test_ids = [str(x) for x in test_ids]
        self.topk_indices = topk_indices  # [N, K] (CPU)
        self.train_ids = train_ids
        self.train_id2desc = train_id2desc
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.topk = topk

        assert len(self.test_ids) == self.topk_indices.size(0), "IDs and retrieved indices mismatch"

    def __len__(self):
        return len(self.test_ids)

    def __getitem__(self, i):
        tid = self.test_ids[i]
        inds = self.topk_indices[i].tolist()
        cand_ids = [self.train_ids[j] for j in inds]
        candidates = [self.train_id2desc.get(str(cid), "") for cid in cand_ids]
        inp = make_rewrite_input(candidates, max_cands=min(self.topk, len(candidates)))

        enc = self.tokenizer(
            inp,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "ID": tid
        }


class PadCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        ids = [b["ID"] for b in batch]

        enc = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt"
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "IDs": ids}


@torch.no_grad()
def generate_captions(t5_model, tokenizer, test_rewrite_dl, device, max_target_len=128, num_beams=2):
    t5_model.eval()
    all_ids, all_caps = [], []

    for batch in test_rewrite_dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        gen = t5_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_target_len,
            num_beams=num_beams
        )
        caps = tokenizer.batch_decode(gen, skip_special_tokens=True)

        all_ids.extend(batch["IDs"])
        all_caps.extend(caps)

    return all_ids, all_caps


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission (Retrieve TopK + T5)")
    parser.add_argument("--env", type=str, default="local", choices=["local", "colab"])

    # Graph encoder params (must match training)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)

    # Checkpoints
    parser.add_argument("--graph_ckpt", type=str, required=True, help="contrastive graph encoder state_dict (.pt)")
    parser.add_argument("--t5_name", type=str, default="t5-small")
    parser.add_argument("--t5_ckpt", type=str, required=True, help="best_t5.pt (state_dict)")

    # Retrieval / generation
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for encoding+generation")
    parser.add_argument("--retrieval_chunk", type=int, default=1024)
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=2)

    # Output
    parser.add_argument("--output_csv", type=str, default=None)

    args = parser.parse_args()

    base_path = "/content/drive/MyDrive/data" if args.env == "colab" else "data"
    TRAIN_GRAPHS = f"{base_path}/train_graphs.pkl"
    TEST_GRAPHS = f"{base_path}/test_graphs.pkl"
    TRAIN_EMB_CSV = f"{base_path}/train_embeddings.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.output_csv is None:
        args.output_csv = f"{base_path}/submission_retrieveTop{args.topk}_T5.csv"

    # --- Checks
    for p in [TRAIN_GRAPHS, TEST_GRAPHS, TRAIN_EMB_CSV, args.graph_ckpt, args.t5_ckpt]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    print(f"Env: {args.env}")
    print(f"Device: {device}")
    print(f"TopK: {args.topk}")
    print(f"Graph ckpt: {args.graph_ckpt}")
    print(f"T5: {args.t5_name} | ckpt: {args.t5_ckpt}")
    print(f"Output: {args.output_csv}")

    # --- Load train text embeddings (for retrieval)
    print("\n📦 Loading train text embeddings...")
    train_emb_dict = load_id2emb(TRAIN_EMB_CSV)
    train_ids = list(train_emb_dict.keys())
    train_txt_embs = _to_tensor_stack([train_emb_dict[i] for i in train_ids], device=device)
    train_txt_embs = F.normalize(train_txt_embs, dim=-1)
    emb_dim = train_txt_embs.size(-1)
    print(f"Train text emb matrix: {train_txt_embs.shape}")

    # --- Load train descriptions
    print("\n📚 Loading train descriptions...")
    train_id2desc = load_descriptions_from_graphs(TRAIN_GRAPHS)
    print(f"Train descriptions: {len(train_id2desc)}")

    # --- Build graph encoder
    print("\n🧠 Loading graph encoder...")
    graph_model = ImprovedMolGraphTransformer(
        hidden=args.hidden,
        out_dim=emb_dim,
        layers=args.layers,
        heads=args.heads
    ).to(device)
    graph_model.load_state_dict(torch.load(args.graph_ckpt, map_location=device))
    graph_model.eval()

    # --- Build test dataset
    print("\n🧪 Loading test dataset...")
    test_ds = PreprocessedGraphDataset(TEST_GRAPHS)  # no text emb for test
    if not hasattr(test_ds, "ids"):
        raise AttributeError("PreprocessedGraphDataset must expose 'ids' for stable submission ordering.")
    test_ids = [str(x) for x in test_ds.ids]
    print(f"Test size: {len(test_ds)}")

    # --- Encode test molecules
    print("\n🔎 Encoding test molecules...")
    test_mol_embs = encode_molecules(graph_model, test_ds, device=device, batch_size=args.batch_size)
    test_mol_embs = F.normalize(test_mol_embs, dim=-1)
    print(f"Test mol emb matrix: {test_mol_embs.shape}")

    # --- Retrieve topK train captions
    print("\n🧲 Retrieving top-K candidates...")
    topk_idx = batched_topk_retrieval(
        mol_embs=test_mol_embs,
        train_txt_embs=train_txt_embs,
        topk=args.topk,
        chunk_size=args.retrieval_chunk
    )
    print(f"TopK indices: {topk_idx.shape}")

    # --- Load T5 rewriter
    print("\n🪶 Loading T5 rewriter...")
    tokenizer = AutoTokenizer.from_pretrained(args.t5_name)
    t5 = AutoModelForSeq2SeqLM.from_pretrained(args.t5_name).to(device)
    t5.load_state_dict(torch.load(args.t5_ckpt, map_location=device))
    t5.eval()

    # --- Build generation dataset/dataloader
    print("\n✍️ Building rewrite inputs + generating captions...")
    test_rewrite_ds = TestRewriteDataset(
        test_ids=test_ids,
        topk_indices=topk_idx,
        train_ids=train_ids,
        train_id2desc=train_id2desc,
        tokenizer=tokenizer,
        max_input_len=args.max_input_len,
        topk=args.topk
    )
    test_rewrite_dl = DataLoader(
        test_rewrite_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PadCollator(tokenizer)
    )

    out_ids, out_caps = generate_captions(
        t5_model=t5,
        tokenizer=tokenizer,
        test_rewrite_dl=test_rewrite_dl,
        device=device,
        max_target_len=args.max_target_len,
        num_beams=args.num_beams
    )

    # --- Save CSV
    df = pd.DataFrame({"ID": out_ids, "description": out_caps})
    df.to_csv(args.output_csv, index=False)
    print("\n" + "=" * 80)
    print(f"✅ Submission saved to: {args.output_csv}")
    print("=" * 80)
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
