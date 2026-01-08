"""
Script pour convertir un modèle PyTorch au format TorchScript (.pt/.jit)
"""

import torch
import torch.nn as nn
import os
import argparse
from networks_2100078 import *


class ModelWrapper(nn.Module):
    """Wrapper pour rendre les modèles compatibles avec TorchScript"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        
    def forward(self, x):
        # Appeler le modèle directement - il gère déjà le preprocessing en interne
        return self.model(x)


def convert_model_to_torchscript(model_path, output_path=None, input_size=None, trace=True):
    """
    Convertit un modèle PyTorch sauvegardé au format TorchScript.
    
    Args:
        model_path (str): Chemin vers le fichier .pth du modèle
        output_path (str): Chemin de sortie pour le modèle TorchScript (optionnel)
        input_size (tuple): Taille de l'exemple d'entrée pour le tracing (optionnel, auto-détecté si None)
        trace (bool): Si True, utilise torch.jit.trace, sinon utilise torch.jit.script
    
    Returns:
        str: Chemin du fichier TorchScript généré
    """
    
    # Charger le modèle
    print(f"Chargement du modèle depuis: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Vérifier si le checkpoint est déjà un modèle
    if isinstance(checkpoint, nn.Module):
        model_type_name = type(checkpoint).__name__
        print(f"Le fichier contient directement un modèle de type: {model_type_name}")
        model = checkpoint
        model.eval()
        
        # Détecter automatiquement la taille d'entrée si non spécifiée
        if input_size is None:
            if 'LSTM' in model_type_name:
                # Pour les LSTM: (batch, seq_len, height, width)
                # Sera transformé en (batch, seq_len, height*width) par le modèle
                input_size = (1, 1, 8, 8)
                print(f"Auto-détection: Modèle LSTM - Taille d'entrée: {input_size}")
            elif 'CNN' in model_type_name:
                # Pour les CNN: (batch, channels, height, width)
                input_size = (1, 1, 8, 8)
                print(f"Auto-détection: Modèle CNN - Taille d'entrée: {input_size}")
            else:
                # Pour les MLP: (batch, features) où features = 64
                input_size = (1, 64)
                print(f"Auto-détection: Modèle MLP - Taille d'entrée: {input_size}")
    else:
        # Extraire le modèle du checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                model_state = checkpoint['state_dict']
            else:
                model_state = checkpoint
        else:
            print("Format de checkpoint non reconnu")
            return None
        
        # Créer une instance du modèle
        # Note: Vous devrez peut-être adapter cela en fonction de votre configuration
        conf = {
            "board_size": 8,
            "path_save": "save_models",
            "earlyStopping": 10,
            "len_inpout_seq": 1,
            "dropout": 0.0
        }
        
        # Essayer de détecter le type de modèle à partir des clés
        if 'conv1.weight' in model_state or any('conv' in k for k in model_state.keys()):
            print("Détection d'un modèle CNN")
            model = CNN_32(conf)
            if input_size is None:
                input_size = (1, 1, 8, 8)
        elif 'lstm' in str(model_state.keys()).lower():
            print("Détection d'un modèle LSTM")
            model = LSTMHiddenState_64(conf)
            if input_size is None:
                input_size = (1, 1, 64)
        else:
            print("Détection d'un modèle MLP")
            model = MLP(conf)
            if input_size is None:
                input_size = (1, 64)
        
        # Charger les poids
        try:
            model.load_state_dict(model_state)
        except Exception as e:
            print(f"Erreur lors du chargement des poids: {e}")
            print("Essai avec strict=False...")
            model.load_state_dict(model_state, strict=False)
        
        model.eval()
    
    # Générer le chemin de sortie si non spécifié
    if output_path is None:
        base_name = os.path.splitext(model_path)[0]
        output_path = f"{base_name}_torchscript.pt"
    
    print(f"Conversion du modèle en TorchScript...")
    print(f"Taille d'entrée utilisée: {input_size}")
    
    try:
        # Créer un exemple d'entrée
        example_input = torch.randn(*input_size)
        
        # Mettre le modèle en mode évaluation
        model.eval()
        
        # Sauvegarder d'abord les poids (toujours possible)
        weights_path = output_path.replace('.pt', '_weights.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': type(model).__name__,
            'input_size': input_size,
            'board_size': 8
        }, weights_path)
        print(f"✓ Poids du modèle sauvegardés à: {weights_path}")
        
        # Test que le modèle fonctionne
        with torch.no_grad():
            try:
                output = model(example_input)
                print(f"✓ Test de forward pass réussi - Sortie: {output.shape}")
            except Exception as e:
                print(f"⚠ Erreur lors du test forward avec forme {input_size}: {e}")
                # Essayer avec différentes formes d'entrée
                if 'LSTM' in type(model).__name__:
                    # Essayer plusieurs formes possibles pour LSTM
                    # Les LSTM avec np.squeeze peuvent avoir besoin de formes spécifiques
                    possible_shapes = [
                        (2, 1, 8, 8),    # (batch>1, seq_len, height, width)
                        (1, 2, 8, 8),    # (batch, seq_len>1, height, width)
                        (2, 8, 8),       # (batch>1, height, width)
                    ]
                    for shape in possible_shapes:
                        try:
                            print(f"Essai avec forme {shape}...")
                            example_input = torch.randn(*shape)
                            output = model(example_input)
                            print(f"✓ Test de forward pass réussi - Forme: {shape} - Sortie: {output.shape}")
                            input_size = shape
                            # Mettre à jour le fichier de poids avec la bonne forme
                            torch.save({
                                'model_state_dict': model.state_dict(),
                                'model_class': type(model).__name__,
                                'input_size': input_size,
                                'board_size': 8
                            }, weights_path)
                            break
                        except Exception as shape_error:
                            continue
                else:
                    print(f"Le modèle ne peut pas être testé, mais les poids sont sauvegardés.")
        
        # Essayer TorchScript trace (peut échouer avec numpy)
        if trace:
            try:
                print("\nTentative de tracing TorchScript (peut échouer avec numpy)...")
                with torch.no_grad():
                    traced_model = torch.jit.trace(model, example_input, strict=False)
                traced_model.save(output_path)
                print(f"✓ Modèle TorchScript (traced) sauvegardé à: {output_path}")
                
                # Vérifier que le modèle peut être rechargé
                print("Vérification du modèle TorchScript...")
                loaded_model = torch.jit.load(output_path)
                with torch.no_grad():
                    test_output = loaded_model(example_input)
                print(f"✓ Modèle TorchScript vérifié - Sortie: {test_output.shape}")
                return output_path
            except Exception as trace_error:
                print(f"⚠ Tracing TorchScript échoué: {trace_error}")
                print(f"\n📌 Les modèles avec numpy ne peuvent pas être convertis en TorchScript.")
                print(f"📌 Utilisez le fichier de poids: {weights_path}")
                return weights_path
        else:
            print("\n📌 Note: torch.jit.script n'est pas supporté pour ces modèles (contiennent du code numpy)")
            print(f"📌 Utilisez le fichier de poids: {weights_path}")
            return weights_path
        
        # Vérifier que le modèle peut être rechargé
        print("Vérification du modèle TorchScript...")
        loaded_model = torch.jit.load(output_path)
        print("✓ Modèle TorchScript chargé avec succès!")
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la conversion: {e}")
        print(f"\n📌 Note: Les modèles contenant du code numpy ne peuvent généralement pas être convertis en TorchScript.")
        # Essayer de sauvegarder au moins les poids
        try:
            weights_path = output_path.replace('.pt', '_weights.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': type(model).__name__,
                'board_size': 8
            }, weights_path)
            print(f"✓ Poids du modèle sauvegardés à: {weights_path}")
            return weights_path
        except:
            return None


def convert_directory(directory_path, output_dir=None, input_size=None):
    """
    Convertit tous les modèles .pth d'un répertoire en TorchScript.
    
    Args:
        directory_path (str): Chemin du répertoire contenant les modèles
        output_dir (str): Répertoire de sortie (optionnel)
        input_size (tuple): Taille de l'exemple d'entrée pour le tracing (optionnel, auto-détecté si None)
    """
    
    if output_dir is None:
        output_dir = os.path.join(directory_path, "torchscript_models")
    
    os.makedirs(output_dir, exist_ok=True)
    
    converted_count = 0
    failed_count = 0
    
    # Parcourir tous les fichiers .pth du répertoire
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pth'):
                model_path = os.path.join(root, file)
                output_name = os.path.splitext(file)[0] + "_torchscript.pt"
                output_path = os.path.join(output_dir, output_name)
                
                print(f"\n{'='*60}")
                result = convert_model_to_torchscript(model_path, output_path, input_size)
                
                if result:
                    converted_count += 1
                else:
                    failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion terminée!")
    print(f"Modèles convertis avec succès: {converted_count}")
    print(f"Échecs: {failed_count}")


def main():
    parser = argparse.ArgumentParser(description='Convertir un modèle PyTorch au format TorchScript')
    parser.add_argument('--model', type=str, help='Chemin vers le fichier .pth du modèle')
    parser.add_argument('--directory', type=str, help='Chemin vers un répertoire contenant des modèles')
    parser.add_argument('--output', type=str, help='Chemin de sortie pour le modèle TorchScript')
    parser.add_argument('--input-size', type=str, default=None, 
                        help='Taille de l\'entrée (format: batch,height,width ou batch,channels,height,width). Auto-détecté si non spécifié.')
    parser.add_argument('--method', type=str, choices=['trace', 'script'], default='trace',
                        help='Méthode de conversion (trace ou script)')
    
    args = parser.parse_args()
    
    # Parser la taille d'entrée
    if args.input_size:
        input_size = tuple(map(int, args.input_size.split(',')))
    else:
        input_size = None
    trace = (args.method == 'trace')
    
    if args.directory:
        # Convertir tous les modèles d'un répertoire
        convert_directory(args.directory, args.output, input_size)
    elif args.model:
        # Convertir un seul modèle
        convert_model_to_torchscript(args.model, args.output, input_size, trace)
    else:
        print("Usage:")
        print("  Convertir un seul modèle:")
        print("    python convert_to_torchscript.py --model chemin/vers/model.pth")
        print("\n  Convertir un répertoire:")
        print("    python convert_to_torchscript.py --directory chemin/vers/repertoire")
        print("\nExemples:")
        print("  python convert_to_torchscript.py --model save_models_MLP/best_model.pth")
        print("  python convert_to_torchscript.py --directory save_models_MLP --output torchscript_models")
        print("  python convert_to_torchscript.py --model model.pth --input-size 1,8,8 --method script")


if __name__ == "__main__":
    main()
