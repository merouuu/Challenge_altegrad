# BLEU-4 Retrieval Integration Summary

## 📋 Overview
Added retrieval-based BLEU-4 evaluation to `train_gt_contrast.py` to align validation metrics with Kaggle scoring. This enables monitoring actual text retrieval performance (BLEU-4) alongside embedding-space metrics (MRR, R@k).

## 🎯 Changes Made

### 1. New Functions Added

#### `load_id2text(graph_path, id_col="id", text_col="description")`
- **Location:** Lines 35-61 in train_gt_contrast.py
- **Purpose:** Loads molecule descriptions from graphs for BLEU evaluation
- **Behavior:**
  - Extracts text descriptions from graph objects
  - Maps graph IDs to description text
  - Returns dict: `{graph_id: description_text}`
  - Robust to missing descriptions (skips with warning)

#### `simple_tokenize(s: str)`
- **Location:** Lines 239-245 in train_gt_contrast.py
- **Purpose:** Safe tokenization for BLEU-4 scoring
- **Behavior:**
  - Uses regex to split on whitespace and punctuation
  - Preserves special tokens (numbers, hyphens)
  - Filters empty tokens

#### `compute_bleu4(pred_texts, ref_texts)`
- **Location:** Lines 246-283 in train_gt_contrast.py
- **Purpose:** Computes corpus-level BLEU-4 score
- **Behavior:**
  - Tokenizes predictions and references
  - Computes NLTK BLEU-4 score with smoothing
  - Handles edge cases (empty sequences)
  - Returns single BLEU-4 score (0.0-1.0)

#### `eval_bleu_retrieval(val_graphs_path, val_emb_dict, val_text_dict, train_emb_dict, train_text_dict, model, device, batch_size=64)`
- **Location:** Lines 286-360 in train_gt_contrast.py
- **Purpose:** Evaluates retrieval-based BLEU-4
- **Algorithm:**
  1. Loads validation graphs and encodes them with the model
  2. For each validation molecule:
     - Computes similarity to all training text embeddings
     - Retrieves top-1 training caption (highest cosine similarity)
  3. Compares retrieved caption to reference using BLEU-4
  4. Returns aggregated BLEU-4 score
- **Key Feature:** Uses `dataset.ids` to ensure correct graph ID alignment

### 2. Integration in Main Training Loop

#### Text Loading (main() function, Lines 722-725)
```python
train_text_dict = load_id2text(TRAIN_GRAPHS) if args.eval_bleu_bert else {}
val_text_dict = load_id2text(VAL_GRAPHS) if args.eval_bleu_bert and os.path.exists(VAL_GRAPHS) else {}
```

#### BLEU-4 Evaluation (main() function, Lines 944-954)
```python
# Évaluer BLEU-4 si activé
if args.eval_bleu_bert and train_text_dict and val_text_dict:
    bleu_results = eval_bleu_retrieval(
        VAL_GRAPHS, val_emb, val_text_dict,
        train_emb, train_text_dict,
        model, DEVICE, batch_size=args.batch_size
    )
    val_metrics.update(bleu_results)
```

#### Logging (main() function, Line 992)
```python
"val_bleu4": float(val_metrics.get("BLEU4", 0.0))
```

#### Display (main() function, Lines 999-1001)
```python
if args.eval_bleu_bert and val_metrics.get("BLEU4"):
    val_str += f", BLEU-4: {val_metrics.get('BLEU4', 0.0):.4f}"
```

## 📊 Expected Output

Training logs will now include:
```json
{
  "epoch": 5,
  "train_loss": 0.1234,
  "val_loss": 0.1567,
  "val_mrr": 0.4521,
  "val_r1": 0.2891,
  "val_r5": 0.5234,
  "val_r10": 0.6789,
  "val_bleu4": 0.1876
}
```

Console output will display:
```
Epoch 05/100 | Train Loss: 0.1234 | Val: Loss: 0.1567, MRR: 0.4521, R@1: 0.2891, BLEU-4: 0.1876
```

## 🔧 Configuration

Enable BLEU-4 evaluation with:
```bash
python train_gt_contrast.py --eval_bleu_bert --batch_size 64
```

Key flags:
- `--eval_bleu_bert`: Enable BLEU-4 + BERTScore evaluation
- `--batch_size`: Batch size for inference (affects memory usage)

## ✅ Data Flow

```
TRAINING PHASE:
├─ train_text_dict = load_id2text(TRAIN_GRAPHS)
└─ train_emb_dict = load_id2emb(TRAIN_EMB_CSV)

VALIDATION PHASE:
├─ val_text_dict = load_id2text(VAL_GRAPHS)
├─ val_emb_dict = load_id2emb(VAL_EMB_CSV)
├─ Encode val molecules with model → mol_emb
├─ Retrieve best train caption via similarity
├─ Compare with reference using compute_bleu4()
└─ Log BLEU-4 score to epoch_log

LOGGING:
├─ training_logs.json: Contains val_bleu4 per epoch
└─ Console: Display BLEU-4 in epoch summary
```

## 🔍 Key Dependency: Dataset ID Alignment

The eval_bleu_retrieval function relies on `PreprocessedGraphDataset.ids` (verified in data_utils.py, line 115):

```python
self.ids = [g.id for g in self.graphs]
```

This ensures graph IDs are accessed in the correct order (pickle load order), matching the embedding dictionary keys.

## 📈 Performance Expectations

- **Initial BLEU-4:** ~0.10-0.15 (random retrieval baseline)
- **After Hard Negatives:** ~0.18-0.25 (semantic clustering improves similarity)
- **Final (epochs 30+):** ~0.25-0.35 (with continued training)

Compare against:
- **MRR:** Should correlate but may lead by 0.1-0.2 (embedding space metric)
- **BERTScore:** Higher than BLEU-4 (semantic similarity is more flexible)

## 🐛 Debugging

If BLEU-4 shows 0.0 consistently:
1. Check that `--eval_bleu_bert` flag is set
2. Verify `load_id2text()` successfully loaded descriptions
3. Confirm `dataset.ids` attribute exists in PreprocessedGraphDataset
4. Check that text embeddings are being loaded correctly

If BLEU-4 is NaN:
- Check for empty text descriptions in graphs
- Verify tokenization handles special characters
- Ensure training text descriptions are valid

## 📝 Files Modified

- `train_gt_contrast.py`: Added 4 functions, integrated into main training loop (3 additional calls, 1 logging line)

## 🎓 Next Steps

1. Run training with `--eval_bleu_bert` to collect BLEU-4 curves
2. Compare BLEU-4 vs MRR trajectories to verify hard negatives improve text retrieval
3. Adjust curriculum learning epoch based on BLEU-4 improvement patterns
4. (Optional) Optimize retrieval strategy (e.g., top-k averaging, reranking)
