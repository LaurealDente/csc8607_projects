"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

def mini_grid_search(
    model_class: Callable,
    get_optimizer_fn: Callable,
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    hparams_grid: Dict[str, List],
    base_model_config: Dict,
    num_epochs: int,
    device: torch.device,
    seed: int = 42,
    log_dir_base: str = "runs/grid_search"
):
    """
    Effectue une mini grid search complète, corrigée pour la stabilité,
    avec suivi, barre de progression, logging TensorBoard et affichage d'un
    tableau récapitulatif final.
    """
    print(f"[INFO] Lancement de la mini grid search pour {num_epochs} époques...")
    
    keys, values = zip(*hparams_grid.items())
    hparam_combinations = [dict(zip(keys, v)) for v in product(*values)]
    print(f"[INFO] {len(hparam_combinations)} combinaisons à tester.")

    results_for_table = []
    best_hparams = None
    best_val_accuracy = -1.0

    for i, hparams in enumerate(hparam_combinations):
        # --- Préparation de l'essai ---
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model_hparams = hparams.copy()
        lr = model_hparams.pop('lr')
        weight_decay = model_hparams.pop('weight_decay')
        
        if 'block_config' in model_hparams:
            B1, B2, B3 = model_hparams.pop('block_config')
            model_hparams.update({'B1': B1, 'B2': B2, 'B3': B3})
        
        current_model_config = {**base_model_config, **model_hparams}
        
        run_name_parts = [f"{k}={v}" for k, v in hparams.items()]
        run_name = "run_" + "_".join(run_name_parts).replace(" ", "").replace("(", "").replace(")", "").replace(",", "-")
        print(f"\n[TEST {i+1}/{len(hparam_combinations)}] : {run_name}")

        model = model_class(**current_model_config).to(device)
        optimizer = get_optimizer_fn(model, weight_decay, lr)
        writer = SummaryWriter(f"{log_dir_base}/{run_name}")
        
        epoch_iterator = tqdm(range(num_epochs), desc="Entraînement")
        training_failed = False
        
        for epoch in epoch_iterator:
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # CORRECTION 1: Vérifier si la perte devient NaN
                if torch.isnan(loss):
                    print(f"\n[ERREUR] Loss est devenue NaN à l'époque {epoch}. Arrêt de cet essai.")
                    training_failed = True
                    break # Sort de la boucle des batches
                
                loss.backward()

                # CORRECTION 2: Ajout du Gradient Clipping pour prévenir l'explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            if training_failed:
                break # Sort de la boucle des époques

            # ... (Le reste de la boucle de validation et de logging reste similaire) ...
            model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            # CORRECTION 3: Prévenir la division par zéro
            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('nan')
            val_accuracy = 100 * correct / total if total > 0 else 0.0
            
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

            epoch_iterator.set_postfix(
                train_loss=f"{avg_train_loss:.4f}", 
                val_loss=f"{avg_val_loss:.4f}", 
                val_acc=f"{val_accuracy:.2f}%"
            )

        # --- Enregistrement des résultats de l'essai ---
        final_val_accuracy = val_accuracy if not training_failed else 0.0
        final_avg_val_loss = avg_val_loss if not training_failed else float('nan')
        notes = "Échoué (NaN)" if training_failed else ""

        if not training_failed and final_val_accuracy > best_val_accuracy:
            best_val_accuracy = final_val_accuracy
            best_hparams = hparams
            print(f"  -> Nouveau meilleur score trouvé : {best_val_accuracy:.2f}%")

        run_summary = {
            'Run': run_name, 'LR': hparams.get('lr'), 'WD': hparams.get('weight_decay'),
            'block_config': str(hparams.get('block_config', 'N/A')), 
            'dropout_p': hparams.get('dropout_p', 'N/A'),
            'Val Acc (%)': final_val_accuracy, 'Val Loss': final_avg_val_loss, 'Notes': notes
        }
        results_for_table.append(run_summary)
        
        writer.add_hparams({k: str(v) for k, v in hparams.items()}, 
                           {'hparam/validation_accuracy': final_val_accuracy, 
                            'hparam/validation_loss': final_avg_val_loss})
        writer.close()

    # --- Affichage du rapport final ---
    print("\n\n" + "="*80)
    print("TABLEAU RÉCAPITULATIF DE LA GRID SEARCH")
    print("="*80)
    
    if results_for_table:
        df = pd.DataFrame(results_for_table)
        df['Val Acc (%)'] = df['Val Acc (%)'].apply(lambda x: f"{x:.2f}")
        df['Val Loss'] = df['Val Loss'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "NaN")
        df = df.sort_values(by='Val Acc (%)', ascending=False)
        print(df.to_markdown(index=False))
    else:
        print("Aucun résultat n'a pu être collecté.")

    print("\n" + "="*50)
    print("MEILLEUR RÉSULTAT TROUVÉ")
    print("="*50)
    if best_hparams:
        print(f"🏆 Meilleure accuracy de validation : {best_val_accuracy:.2f}%")
        print("Hyperparamètres correspondants :")
        for key, value in best_hparams.items():
            print(f"  - {key}: {value}")
    else:
        print("Aucun essai n'a réussi à produire un résultat valide.")
    print("="*50)
    print(f"\n[INFO] Grid search terminée. Lancez 'tensorboard --logdir {log_dir_base}' pour une analyse détaillée.")




def set_seed(seed: int) -> None:
    """Initialise les seeds (numpy/torch/python). À implémenter."""
    raise NotImplementedError("set_seed doit être implémentée par l'étudiant·e.")


def get_device(prefer: str | None = "auto") -> str:
    """Retourne 'cpu' ou 'cuda' (ou choix basé sur 'auto'). À implémenter."""
    raise NotImplementedError("get_device doit être implémentée par l'étudiant·e.")


def count_parameters(model) -> int:
    """Retourne le nombre de paramètres entraînables du modèle. À implémenter."""
    raise NotImplementedError("count_parameters doit être implémentée par l'étudiant·e.")


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """Sauvegarde une copie de la config (ex: YAML) dans out_dir. À implémenter."""
    raise NotImplementedError("save_config_snapshot doit être implémentée par l'étudiant·e.")