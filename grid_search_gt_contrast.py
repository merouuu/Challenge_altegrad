"""
Grid Search Script for Graph Transformer with Contrastive Loss
Optimisé pour Google Colab

Ce script lance plusieurs entraînements avec différentes combinaisons d'hyperparamètres
et génère un résumé comparatif à la fin.
"""

import os
import json
import itertools
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# =========================================================
# CONFIGURATION DE LA GRID SEARCH
# =========================================================

# Environnement (colab ou local)
ENV = "colab"  # Changez en "local" si vous testez localement

# Base path selon l'environnement
BASE_PATH = "/content/drive/MyDrive/data" if ENV == "colab" else "data"

# Grille d'hyperparamètres à tester
GRID_SEARCH_CONFIG = {
    "lr": [0.0001, 0.0003, 0.0005],  # Learning rates à tester
    "temp": [0.05, 0.07, 0.1],       # Températures pour la loss contrastive
    "hidden": [128, 256],            # Dimensions cachées
    "layers": [3, 4, 5],             # Nombre de couches transformer
    "heads": [4, 8],                 # Nombre de têtes d'attention
    "batch_size": [32, 64],          # Tailles de batch
}

# Paramètres fixes pour tous les runs
FIXED_PARAMS = {
    "epochs": 20,  # Nombre d'epochs par run
    "env": ENV,
}

# Option pour reprendre après interruption (ignore les runs déjà complétés)
RESUME_MODE = True  # Mettre à False pour forcer la réexécution de tous les runs

# =========================================================
# FONCTIONS UTILITAIRES
# =========================================================

def generate_run_id(config):
    """Génère un ID unique pour un run basé sur sa configuration."""
    parts = []
    for key in sorted(config.keys()):
        if key != "epochs" and key != "env":
            value = config[key]
            # Format compact pour les valeurs
            if isinstance(value, float):
                parts.append(f"{key}_{value:.4f}".replace(".", "p"))
            else:
                parts.append(f"{key}_{value}")
    return "_".join(parts)


def run_training(config, run_id):
    """Lance un entraînement avec une configuration donnée."""
    print(f"\n{'='*80}")
    print(f"🚀 Démarrage du run: {run_id}")
    print(f"{'='*80}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    print(f"{'='*80}\n")
    
    # Construction de la commande
    cmd = [
        sys.executable,
        "train_gt_contrast.py",
        "--run_id", run_id,
        "--env", config["env"],
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"]),
        "--temp", str(config["temp"]),
        "--hidden", str(config["hidden"]),
        "--layers", str(config["layers"]),
        "--heads", str(config["heads"]),
        "--batch_size", str(config["batch_size"]),
    ]
    
    # Exécution
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Afficher la sortie en temps réel
            text=True
        )
        print(f"\n✅ Run {run_id} terminé avec succès\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur lors du run {run_id}: {e}\n")
        return False


def is_run_completed(run_id):
    """Vérifie si un run est déjà complété."""
    logs_path = f"{BASE_PATH}/GT_Contrast/run_{run_id}/training_logs.json"
    if not os.path.exists(logs_path):
        return False
    
    try:
        with open(logs_path, 'r') as f:
            logs = json.load(f)
        
        # Vérifier si le run a atteint le nombre d'epochs prévu
        epochs = logs.get("epochs", [])
        config = logs.get("config", {})
        expected_epochs = config.get("epochs", FIXED_PARAMS["epochs"])
        
        # Le run est complété s'il a au moins le nombre d'epochs attendu
        return len(epochs) >= expected_epochs
    except:
        return False


def load_run_results(run_id):
    """Charge les résultats d'un run depuis son fichier de logs."""
    logs_path = f"{BASE_PATH}/GT_Contrast/run_{run_id}/training_logs.json"
    
    if not os.path.exists(logs_path):
        return None
    
    try:
        with open(logs_path, 'r') as f:
            logs = json.load(f)
        
        # Extraire les meilleures métriques
        best_mrr = logs.get("best_mrr", 0.0)
        config = logs.get("config", {})
        
        # Trouver le meilleur epoch
        epochs = logs.get("epochs", [])
        if epochs:
            best_epoch = max(epochs, key=lambda x: x.get("val_mrr", 0.0))
            return {
                "run_id": run_id,
                "config": config,
                "best_mrr": best_mrr,
                "best_epoch": best_epoch.get("epoch", 0),
                "best_r1": best_epoch.get("val_r1", 0.0),
                "best_r5": best_epoch.get("val_r5", 0.0),
                "best_r10": best_epoch.get("val_r10", 0.0),
                "final_train_loss": epochs[-1].get("train_loss", 0.0) if epochs else 0.0,
            }
        else:
            return {
                "run_id": run_id,
                "config": config,
                "best_mrr": best_mrr,
                "best_epoch": 0,
                "best_r1": 0.0,
                "best_r5": 0.0,
                "best_r10": 0.0,
                "final_train_loss": 0.0,
            }
    except Exception as e:
        print(f"⚠️  Erreur lors du chargement des résultats pour {run_id}: {e}")
        return None


def generate_summary(all_results):
    """Génère un résumé comparatif de tous les runs."""
    if not all_results:
        return "Aucun résultat à afficher."
    
    # Trier par MRR décroissant
    sorted_results = sorted(all_results, key=lambda x: x["best_mrr"], reverse=True)
    
    summary_lines = [
        "\n" + "="*100,
        "📊 RÉSUMÉ DE LA GRID SEARCH",
        "="*100,
        f"\nTotal de runs: {len(all_results)}",
        f"Meilleur MRR: {sorted_results[0]['best_mrr']:.4f}",
        "\n" + "-"*100,
        "TOP 10 MEILLEURS RUNS (par MRR):",
        "-"*100,
        f"\n{'Rank':<6} {'Run ID':<40} {'MRR':<8} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'LR':<10} {'Temp':<8} {'Hidden':<8} {'Layers':<8} {'Heads':<8}",
        "-"*100,
    ]
    
    for rank, result in enumerate(sorted_results[:10], 1):
        config = result["config"]
        summary_lines.append(
            f"{rank:<6} {result['run_id'][:38]:<40} "
            f"{result['best_mrr']:<8.4f} {result['best_r1']:<8.4f} "
            f"{result['best_r5']:<8.4f} {result['best_r10']:<8.4f} "
            f"{config.get('lr', 0):<10.5f} {config.get('temp', 0):<8.3f} "
            f"{config.get('hidden', 0):<8} {config.get('layers', 0):<8} "
            f"{config.get('heads', 0):<8}"
        )
    
    summary_lines.append("\n" + "="*100)
    summary_lines.append("DÉTAILS DE TOUS LES RUNS:")
    summary_lines.append("="*100)
    
    for result in sorted_results:
        config = result["config"]
        summary_lines.append(f"\nRun ID: {result['run_id']}")
        summary_lines.append(f"  Config: lr={config.get('lr')}, temp={config.get('temp')}, "
                           f"hidden={config.get('hidden')}, layers={config.get('layers')}, "
                           f"heads={config.get('heads')}, batch_size={config.get('batch_size')}")
        summary_lines.append(f"  Best MRR: {result['best_mrr']:.4f} (epoch {result['best_epoch']})")
        summary_lines.append(f"  Best R@1: {result['best_r1']:.4f}, R@5: {result['best_r5']:.4f}, "
                           f"R@10: {result['best_r10']:.4f}")
    
    return "\n".join(summary_lines)


# =========================================================
# MAIN GRID SEARCH
# =========================================================

def main():
    print("="*80)
    print("🔬 GRID SEARCH - Graph Transformer with Contrastive Loss")
    print("="*80)
    print(f"Environnement: {ENV}")
    print(f"Base path: {BASE_PATH}")
    print(f"\nGrille de recherche:")
    for key, values in GRID_SEARCH_CONFIG.items():
        print(f"  {key}: {values}")
    print(f"\nParamètres fixes: {FIXED_PARAMS}")
    
    # Générer toutes les combinaisons
    keys = list(GRID_SEARCH_CONFIG.keys())
    values = list(GRID_SEARCH_CONFIG.values())
    all_combinations = list(itertools.product(*values))
    
    total_runs = len(all_combinations)
    print(f"\n📊 Total de combinaisons à tester: {total_runs}")
    print("="*80)
    
    # Demander confirmation
    if ENV == "colab":
        response = input("\n⚠️  Voulez-vous continuer? Cela peut prendre plusieurs heures. (yes/no): ")
        if response.lower() not in ["yes", "y", "oui", "o"]:
            print("Grid search annulée.")
            return
    else:
        print("\n⚠️  Démarrage de la grid search dans 5 secondes... (Ctrl+C pour annuler)")
        import time
        time.sleep(5)
    
    # Créer le dossier de résultats de la grid search
    grid_search_dir = f"{BASE_PATH}/GT_Contrast/grid_search"
    os.makedirs(grid_search_dir, exist_ok=True)
    
    # Charger les résultats intermédiaires existants si en mode resume
    all_results = []
    completed_run_ids = set()
    
    if RESUME_MODE:
        intermediate_path = f"{grid_search_dir}/intermediate_results.json"
        if os.path.exists(intermediate_path):
            try:
                with open(intermediate_path, 'r') as f:
                    intermediate = json.load(f)
                    all_results = intermediate.get("results", [])
                    completed_run_ids = {r["run_id"] for r in all_results}
                    print(f"📂 {len(completed_run_ids)} runs déjà complétés détectés")
            except Exception as e:
                print(f"⚠️  Erreur lors du chargement des résultats intermédiaires: {e}")
    
    # Lancer tous les runs
    successful_runs = len(completed_run_ids)
    failed_runs = 0
    skipped_runs = 0
    
    start_time = datetime.now()
    
    for idx, combination in enumerate(all_combinations, 1):
        # Créer la configuration complète
        config = {key: value for key, value in zip(keys, combination)}
        config.update(FIXED_PARAMS)
        
        # Générer l'ID du run
        run_id = generate_run_id(config)
        
        # Vérifier si le run est déjà complété (en mode resume)
        if RESUME_MODE and is_run_completed(run_id):
            print(f"\n[{idx}/{total_runs}] Run ID: {run_id} - ⏭️  Déjà complété, ignoré")
            skipped_runs += 1
            # Charger les résultats si pas déjà chargés
            if run_id not in completed_run_ids:
                results = load_run_results(run_id)
                if results:
                    all_results.append(results)
                    completed_run_ids.add(run_id)
            continue
        
        print(f"\n[{idx}/{total_runs}] Run ID: {run_id}")
        
        # Lancer l'entraînement
        success = run_training(config, run_id)
        
        if success:
            # Charger les résultats
            results = load_run_results(run_id)
            if results:
                all_results.append(results)
                successful_runs += 1
            else:
                failed_runs += 1
                print(f"⚠️  Impossible de charger les résultats pour {run_id}")
        else:
            failed_runs += 1
        
        # Sauvegarder les résultats intermédiaires
        intermediate_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": total_runs,
            "completed": idx,
            "successful": successful_runs,
            "failed": failed_runs,
            "results": all_results
        }
        with open(f"{grid_search_dir}/intermediate_results.json", 'w') as f:
            json.dump(intermediate_summary, f, indent=2)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Générer le résumé final
    summary = generate_summary(all_results)
    print(summary)
    
    # Sauvegarder le résumé
    summary_path = f"{grid_search_dir}/grid_search_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    # Sauvegarder les résultats complets en JSON
    final_results = {
        "grid_search_config": GRID_SEARCH_CONFIG,
        "fixed_params": FIXED_PARAMS,
        "resume_mode": RESUME_MODE,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "skipped_runs": skipped_runs if RESUME_MODE else 0,
        "results": all_results
    }
    
    results_json_path = f"{grid_search_dir}/grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_json_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✅ Grid search terminée!")
    print(f"   Durée totale: {duration}")
    print(f"   Runs réussis: {successful_runs}/{total_runs}")
    print(f"   Runs échoués: {failed_runs}/{total_runs}")
    if RESUME_MODE and skipped_runs > 0:
        print(f"   Runs ignorés (déjà complétés): {skipped_runs}/{total_runs}")
    print(f"\n📁 Résultats sauvegardés dans:")
    print(f"   - Résumé texte: {summary_path}")
    print(f"   - Résultats JSON: {results_json_path}")


if __name__ == "__main__":
    main()
