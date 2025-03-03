import argparse
import yaml
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement complet")
    parser.add_argument("-m", "--model", type=str, required=True, help="Type de modèle (ex: yoloPose)")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Nom du dataset (ex: dataset1)")
    parser.add_argument("-metric", "--metrics", type=str, nargs='+', required=False, default=["MaP"],
                        help="Liste des métriques (ex: MaP accuracy)")
    return parser.parse_args()

def load_config(model, dataset):
    # Construire le nom du fichier de config à partir des arguments
    config_filename = f"{model}_{dataset}.yaml"
    config_path = os.path.join("configs", config_filename)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    args = parse_args()
    
    # Charger la config en fonction des arguments
    config = load_config(args.model, args.dataset)
    
    # Mise à jour éventuelle des métriques avec les arguments passés en CLI
    if args.metrics:
        config['metrics'] = args.metrics
    
    # Affichage pour vérification
    print("Configuration utilisée :")
    print(yaml.dump(config, default_flow_style=False))
    
    # Ici vous pouvez initialiser votre modèle, charger vos données, etc.
    # Exemple pseudo-code :
    # model = init_model(config['model']['type'], **config['model']['parameters'])
    # train_loader, val_loader, test_loader = load_dataset(config['dataset']['path'], config['dataset']['splits'])
    # trainer = Trainer(model, train_loader, val_loader, config['training'], config['metrics'])
    # trainer.train()
    # trainer.evaluate()

if __name__ == "__main__":
    main()
