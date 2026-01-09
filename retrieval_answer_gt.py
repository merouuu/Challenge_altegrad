import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import (
    load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
)

from train_gt import MolGraphTransformer

# Configuration des arguments
parser = argparse.ArgumentParser(description="Générer une soumission Kaggle")
parser.add_argument('--env', type=str, default='local', choices=['local', 'colab'], 
                    help="Définir l'environnement : 'local' ou 'colab'")
args = parser.parse_args()

# Chemins de base
base_path = "/content/drive/MyDrive/data" if args.env == 'colab' else "data"
TRAIN_GRAPHS = f"{base_path}/train_graphs.pkl"
TEST_GRAPHS  = f"{base_path}/test_graphs.pkl"
TRAIN_EMB_CSV = f"{base_path}/GT/train_embeddings.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def retrieve_descriptions(model, train_data, test_data, train_emb_dict, device, output_csv):
    """
    Fonction principale de récupération : 
    1. Calcule les embeddings des graphes de test.
    2. Les compare aux embeddings de texte du train.
    3. Récupère la description la plus proche.
    """
    print("Chargement des descriptions d'entraînement...")
    train_id2desc = load_descriptions_from_graphs(train_data)
    
    # Préparer les embeddings de texte (Train)
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[id_] for id_ in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)
    
    print(f"Base de données de récupération : {len(train_ids)} molécules")
    
    # Préparer les données de Test
    test_ds = PreprocessedGraphDataset(test_data)
    # Important : shuffle=False pour garder l'ordre des IDs
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    print(f"Molécules de test à prédire : {len(test_ds)}")
    
    # Encodage des molécules de Test
    model.eval()
    test_mol_embs = []
    test_ids_ordered = []
    
    print("Encodage des graphes de test en cours...")
    for graphs in test_dl: # Pas de text_emb ici car c'est le test set
        graphs = graphs.to(device)
        mol_emb = model(graphs)
        test_mol_embs.append(mol_emb)
        
        # Récupérer les IDs du batch courant
        # Note: PreprocessedGraphDataset stocke les IDs dans self.ids
        # On doit faire attention à l'alignement, le DataLoader itère séquentiellement
        # donc on peut simplement étendre la liste, mais le dataset doit être chargé complet.
        # Une méthode plus sûre est de récupérer les indices si le collate le permettait,
        # mais ici on va supposer l'ordre séquentiel du dataset.
    
    # Récupération des IDs correspondants (tricky part avec PyG DataLoader)
    # Le dataset est indexable, donc on peut simplement prendre la liste complète
    test_ids_ordered = test_ds.ids 

    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    
    # Calcul de similarité (Produit scalaire car vecteurs normalisés)
    # [N_test, Dim] @ [Dim, N_train] -> [N_test, N_train]
    print("Calcul des similarités...")
    similarities = test_mol_embs @ train_embs.t()
    
    # Trouver l'indice du max pour chaque molécule de test
    most_similar_indices = similarities.argmax(dim=-1).cpu()
    
    # Construction du CSV
    results = []
    for i, test_id in enumerate(test_ids_ordered):
        train_idx = most_similar_indices[i].item()
        retrieved_train_id = train_ids[train_idx]
        retrieved_desc = train_id2desc[retrieved_train_id]
        
        results.append({
            'ID': test_id,
            'description': retrieved_desc
        })
        
        if i < 3: # Afficher quelques exemples
            print(f"\n[Exemple] Test ID {test_id} -> Train ID {retrieved_train_id}")
            print(f"Caption: {retrieved_desc[:100]}...")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Soumission sauvegardée : {output_csv}")
    print(f"{'='*80}")
    
    return results_df

def main():
    print(f"Environnement : {args.env}")
    print(f"Device : {DEVICE}")
    
    # Vérification des fichiers
    if not os.path.exists(TEST_GRAPHS):
        print(f"Erreur: Fichier de test non trouvé : {TEST_GRAPHS}")
        return

    # Chargement des embeddings de texte (nécessaire pour connaitre la dimension et pour la récupération)
    print("Chargement des embeddings TEXTE d'entraînement...")
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    emb_dim = len(next(iter(train_emb.values())))
    
    # Configuration Baseline
    model = MolGraphTransformer(hidden=128, out_dim=emb_dim, layers=4, heads=4).to(DEVICE)
    model_path = f"{base_path}/GT/model_checkpoint.pt"
    output_csv = f"{base_path}/submission_GT.csv"
    
    # Chargement des poids
    if not os.path.exists(model_path):
        print(f"Erreur : Checkpoint non trouvé à {model_path}")
        return

    print(f"Chargement des poids depuis {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # Lancement de la récupération
    retrieve_descriptions(
        model=model,
        train_data=TRAIN_GRAPHS,
        test_data=TEST_GRAPHS,
        train_emb_dict=train_emb,
        device=DEVICE,
        output_csv=output_csv
    )

if __name__ == "__main__":
    main()