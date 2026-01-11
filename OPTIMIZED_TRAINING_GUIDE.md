# Guide d'entraînement optimisé - Corrections critiques appliquées

## 🔴 Corrections critiques appliquées

### 1. **FIX MAJEUR: Désalignement Dataset/Sampler** ✅
**Problème:** Le HardNegativeSampler construisait les embeddings dans l'ordre **lexicographique des IDs**, mais le dataset était dans l'ordre du **pickle**. Résultat : on faisait du batching sur de mauvaises molécules !

**Solution:** Le sampler prend maintenant le dataset en paramètre et utilise son ordre réel.

```python
# AVANT (bug)
sampler = HardNegativeSampler(train_emb, batch_size=32)

# APRÈS (correct)
sampler = HardNegativeSampler(train_emb, batch_size=32, dataset=train_ds)
```

### 2. **AttentionPooling optimisée** ✅
- Remplacé calcul manuel du softmax par `torch_geometric.utils.softmax` (plus stable numériquement)
- Éliminé calcul redondant de `scatter(...).exp()`

### 3. **Graph Augmentation ajoutée** ✅
- Edge dropout : supprime 10% des edges aléatoirement
- Node feature dropout : masque 5% des features
- Force le modèle à apprendre des invariances

### 4. **Temperature Schedule** ✅
- Démarre à 0.1 (softmax plus diffus, moins aiguisé)
- Décroît linéairement jusqu'à 0.07 (plus sélectif en fin)
- Évite sur-spécialisation trop tôt

### 5. **BLEU-4 + BERTScore Evaluation** ✅
- Mesure des métriques que Kaggle utilise réellement
- Permet de piloter en fonction du score final, pas juste MRR

---

## 🚀 Commandes optimales par stratégie

### **Stratégie 1 : Démarrage complet (recommandé)**

```bash
python train_gt_contrast.py \
  --env colab \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.0003 \
  --temp 0.07 \
  --hard_negatives \
  --hard_ratio 0.5 \
  --hardness_k 100 \
  --curriculum_epoch 5 \
  --use_augmentation \
  --temp_schedule
```

**Phases:**
- Epochs 1-5 : Random sampling + warm-up
- Epochs 6-100 : Hard negative mining + temperature decay

**Taux de succès attendu:** MRR 0.35-0.45, BLEU-4 0.15-0.20

### **Stratégie 2 : Curriculum + Évaluation texte**

```bash
python train_gt_contrast.py \
  --env colab \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.0003 \
  --temp 0.07 \
  --hard_negatives \
  --hard_ratio 0.5 \
  --hardness_k 100 \
  --curriculum_epoch 5 \
  --use_augmentation \
  --temp_schedule \
  --eval_bleu_bert
```

**Avantage:** Voit les scores BLEU-4 + BERTScore réels (ce que Kaggle mesure)

### **Stratégie 3 : Hard mode agressif (fine-tuning)**

```bash
python train_gt_contrast.py \
  --env colab \
  --epochs 50 \
  --batch_size 128 \
  --lr 0.0001 \
  --temp 0.07 \
  --hard_negatives \
  --hard_ratio 0.7 \
  --hardness_k 50 \
  --curriculum_epoch 0 \
  --use_augmentation \
  --temp_schedule \
  --resume_from GT_Contrast/contrastive_model.pt
```

**Quand l'utiliser:** Vous reprenez un modèle déjà entraîné (epochs 5+)

---

## 📊 Hyperparamètres clés et recommandations

### Hard Ratio (après curriculum)

| Valeur | Profil | Cas d'usage |
|--------|--------|-----------|
| 0.3 | Conservateur | Démarrage instable, small batch |
| **0.5** | **Équilibré (👈 recommandé)** | **Tous les cas** |
| 0.7 | Agressif | Fine-tuning |
| 0.8+ | Hardcore | Risque de divergence |

### Curriculum Epoch

| Valeur | Effet | Cas d'usage |
|--------|-------|-----------|
| 0 | Hard dès le début | Modèle déjà prétraîné |
| **3-5** | **Recommandé** | **Démarrage cold** |
| 10+ | Trop tard | Modèle a déjà convergé |

### Temperature Schedule

**Activé (recommandé):**
- Démarre 0.1 → Finit 0.07
- Évite sur-aiguisage trop tôt
- +2-3% MRR souvent

**Désactivé:**
- Température fixe à 0.07
- Peut causer oscillation

---

## 🎯 Résultats attendus

### Avec les corrections

| Métrique | Sans hard neg | Avec curriculum | Avec augmentation |
|----------|--------------|-----------------|-------------------|
| **MRR** | 0.30-0.35 | 0.40-0.50 | 0.42-0.52 |
| **R@1** | 0.15-0.20 | 0.25-0.35 | 0.27-0.37 |
| **R@5** | 0.45-0.55 | 0.60-0.70 | 0.62-0.72 |
| **BLEU-4** | 0.10-0.13 | 0.15-0.18 | 0.16-0.20 |

---

## ⚠️ Points critiques

### 1. Dataset alignment ✅
- Le sampler reçoit `dataset=train_ds`
- Les neighbors sont dans l'ordre réel du dataset
- **Vérifier:** Les logs affichent `✅ Utilisation de l'ordre réel du dataset`

### 2. Évaluation texte
- Si `--eval_bleu_bert` : nécessite `nltk` et `bert_score`
- Installation: `pip install nltk bert-score`
- Sinon : juste MRR/R@k (toujours valide)

### 3. Température schedule
- Si activé: température adaptative par epoch
- Si désactivé: fixe à `--temp` (0.07)
- Bénéfice: ~2-3% de gain MRR

### 4. Augmentation
- Edge dropout: 10% des edges
- Node dropout: 5% des features
- Ajoute robustesse, légèrement ralentit chaque epoch

---

## 📈 Monitoring pendant l'entraînement

### Signes positifs ✅

```
Epoch 01/100 | Train Loss: 3.5152 | Val: MRR: 0.1124
Epoch 02/100 | Train Loss: 3.1656 | Val: MRR: 0.1467  ← montée régulière
Epoch 05/100 | Train Loss: 2.9624 | Val: MRR: 0.2012

🎓 CURRICULUM SWITCH: Activation du Hard Negative Mining à l'epoch 6
Epoch 06/100 | Train Loss: 3.0891 | Val: MRR: 0.2045  ← petit pic de loss, MRR continue
Epoch 07/100 | Train Loss: 2.8945 | Val: MRR: 0.2234  ← reprend sa montée
```

### Signes problématiques ⚠️

```
Epoch 10/100 | Train Loss: 2.8000 | Val: MRR: 0.3000
Epoch 11/100 | Train Loss: 2.7500 | Val: MRR: 0.3001  ← stagnation
Epoch 12/100 | Train Loss: 2.8200 | Val: MRR: 0.2987  ← baisse (divergence possible)
```

**Actions:**
1. Réduire `--hard_ratio` (0.5 → 0.3)
2. Réduire `--lr` (0.0003 → 0.0001)
3. Repousser curriculum (5 → 8)

---

## 🔗 Recommandations finales

### Pour soumettre à Kaggle

1. **Entraîner avec la stratégie 1** (curriculum + augmentation)
2. **Valider avec BLEU-4 + BERTScore** (strategy 2)
3. **Si plateauing**: passer en "mode agressif" (strategy 3)
4. **Toujours sauvegarder** `GT_Contrast/contrastive_model.pt`

### Ordre de priorité si RAM limite

1. Garder `--hard_negatives` + curriculum (gain majeur)
2. Augmenter batch_size à 64+ (contrastive loss aime ça)
3. Réduire `--hidden` si OOM (128 → 96)
4. `--use_augmentation` optionnel (gain modéré)

---

## 📝 Notes techniques

- **HardNegativeSampler** : O(N*K) precomp, puis O(batch) per epoch
- **Temperature schedule** : décroissance linéaire `init_temp - (init_temp - final)*progress`
- **Softmax PyG** : stable numériquement même pour gros batches
- **BLEU-4** : requires tokenization (nltk.punkt)
- **BERTScore** : requires pretrained BERT (auto-download)

Bon chance ! 🚀
