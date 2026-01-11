# Guide d'utilisation de la Grid Search

Ce guide explique comment utiliser le script de grid search pour tester différentes combinaisons d'hyperparamètres sur Google Colab.

## 📋 Prérequis

1. **Fichiers nécessaires sur Colab:**
   - `train_gt_contrast.py` (modifié avec support `--run_id`)
   - `grid_search_gt_contrast.py`
   - `data_utils.py`
   - Les données dans `/content/drive/MyDrive/data/`

2. **Structure des données:**
   ```
   /content/drive/MyDrive/data/
   ├── train_graphs.pkl
   ├── validation_graphs.pkl
   ├── train_embeddings.csv
   └── validation_embeddings.csv
   ```

## 🚀 Utilisation sur Google Colab

### Étape 1: Configuration de la Grid Search

Ouvrez `grid_search_gt_contrast.py` et modifiez la section `GRID_SEARCH_CONFIG` selon vos besoins:

```python
GRID_SEARCH_CONFIG = {
    "lr": [0.0001, 0.0003, 0.0005],  # Learning rates à tester
    "temp": [0.05, 0.07, 0.1],       # Températures pour la loss contrastive
    "hidden": [128, 256],            # Dimensions cachées
    "layers": [3, 4, 5],             # Nombre de couches transformer
    "heads": [4, 8],                 # Nombre de têtes d'attention
    "batch_size": [32, 64],          # Tailles de batch
}
```

**⚠️ Attention:** Le nombre total de combinaisons est le produit de toutes les valeurs. 
Par exemple: `3 × 3 × 2 × 3 × 2 × 2 = 108 combinaisons`

### Étape 2: Lancer la Grid Search

Dans une cellule Colab:

```python
!python grid_search_gt_contrast.py
```

Ou si vous préférez voir la sortie en temps réel:

```python
import subprocess
import sys

result = subprocess.run([sys.executable, "grid_search_gt_contrast.py"], 
                       text=True)
```

### Étape 3: Suivre la progression

Le script:
- Affiche la progression en temps réel
- Sauvegarde les résultats intermédiaires dans `GT_Contrast/grid_search/intermediate_results.json`
- Crée un dossier séparé pour chaque run: `GT_Contrast/run_{run_id}/`

### Étape 4: Analyser les résultats

Après la fin de la grid search, les résultats sont sauvegardés dans:
- `GT_Contrast/grid_search/grid_search_summary_{timestamp}.txt` - Résumé textuel
- `GT_Contrast/grid_search/grid_search_results_{timestamp}.json` - Résultats complets en JSON

## 📊 Structure des résultats

Chaque run est sauvegardé dans son propre dossier:
```
GT_Contrast/
├── run_lr_0p0001_temp_0p0500_hidden_128_layers_3_heads_4_batch_size_32/
│   ├── contrastive_model.pt
│   ├── checkpoint.pt
│   └── training_logs.json
├── run_lr_0p0001_temp_0p0500_hidden_128_layers_3_heads_4_batch_size_64/
│   └── ...
└── grid_search/
    ├── intermediate_results.json
    ├── grid_search_summary_20240101_120000.txt
    └── grid_search_results_20240101_120000.json
```

## 🔧 Personnalisation

### Réduire le nombre de combinaisons

Pour tester rapidement, réduisez les listes:

```python
GRID_SEARCH_CONFIG = {
    "lr": [0.0003],           # 1 valeur
    "temp": [0.05, 0.07],     # 2 valeurs
    "hidden": [128],          # 1 valeur
    "layers": [4],            # 1 valeur
    "heads": [4],             # 1 valeur
    "batch_size": [32],       # 1 valeur
}
# Total: 1 × 2 × 1 × 1 × 1 × 1 = 2 combinaisons
```

### Modifier les paramètres fixes

```python
FIXED_PARAMS = {
    "epochs": 10,  # Réduire pour tester plus vite
    "env": "colab",
}
```

### Reprendre après une interruption

Le script sauvegarde les résultats intermédiaires. Vous pouvez:
1. Modifier le script pour ignorer les runs déjà complétés
2. Ou simplement relancer - les runs déjà faits seront écrasés (mais leurs résultats sont dans `intermediate_results.json`)

## 💡 Conseils

1. **Commencez petit:** Testez avec 2-3 combinaisons d'abord pour vérifier que tout fonctionne
2. **Surveillez la RAM:** Les gros modèles peuvent consommer beaucoup de mémoire
3. **Utilisez GPU:** Assurez-vous qu'un GPU est disponible sur Colab
4. **Sauvegardez régulièrement:** Les résultats intermédiaires sont sauvegardés automatiquement

## 📈 Analyser les résultats avec Python

```python
import json
import pandas as pd

# Charger les résultats
with open('/content/drive/MyDrive/data/GT_Contrast/grid_search/grid_search_results_*.json', 'r') as f:
    results = json.load(f)

# Convertir en DataFrame pour analyse
df = pd.DataFrame([
    {
        **r['config'],
        'best_mrr': r['best_mrr'],
        'best_r1': r['best_r1'],
        'best_r5': r['best_r5'],
        'best_r10': r['best_r10'],
    }
    for r in results['results']
])

# Trier par MRR
df_sorted = df.sort_values('best_mrr', ascending=False)
print(df_sorted.head(10))

# Visualiser les corrélations
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Corrélation entre hyperparamètres et performances')
plt.show()
```

## ⚠️ Notes importantes

- Chaque run peut prendre 30 minutes à plusieurs heures selon la configuration
- Le nombre total de runs = produit de toutes les valeurs dans `GRID_SEARCH_CONFIG`
- Les résultats sont sauvegardés automatiquement, mais vérifiez régulièrement l'espace disque
- En cas d'erreur sur un run, le script continue avec les autres runs
