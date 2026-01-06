import h5py
import numpy as np
from tqdm import tqdm
import copy
from utile import has_tile_to_flip, isBlackWinner, initialze_board, BOARD_SIZE
from torch.utils.data import Dataset


def load_game_log(file_path):
    """Charge le log de jeu depuis un fichier HDF5"""
    h5f = h5py.File(file_path, 'r')
    game_name = file_path.split('/')[-1].replace(".h5", "")
    game_log = np.array(h5f[game_name][:])
    h5f.close()
    return game_log


class CustomDatasetAll(Dataset):
    """
    Dataset qui utilise TOUS les échantillons (gagnants ET perdants)
    au lieu de seulement les échantillons du gagnant
    """
    
    def __init__(self, dataset_conf, load_data_once4all=True):
        """
        Custom dataset class pour Othello - Version utilisant toutes les données
        
        Parameters:
        - dataset_conf (dict): Configuration du dataset
        - load_data_once4all (bool): Charger toutes les données en mémoire
        """
        
        self.load_data_once4all = load_data_once4all
        self.starting_board_stat = initialze_board()
        
        # Configuration
        self.filelist = dataset_conf["filelist"]
        self.len_samples = dataset_conf["len_samples"]
        self.path_dataset = dataset_conf["path_dataset"]
        self.use_all_samples = dataset_conf.get("use_all_samples", True)
        
        # Lire les noms de fichiers
        with open(self.filelist) as f:
            list_files = [line.rstrip() for line in f]
        self.game_files_name = list_files
        
        if self.load_data_once4all:
            # Calculer le nombre total d'échantillons
            # Si use_all_samples=True: 60 samples par jeu (30 pour chaque joueur)
            # Si use_all_samples=False: 30 samples par jeu (seulement le gagnant)
            samples_per_game = 60 if self.use_all_samples else 30
            
            self.samples = np.zeros(
                (len(self.game_files_name) * samples_per_game, self.len_samples, 8, 8),
                dtype=int
            )
            self.outputs = np.zeros(
                (len(self.game_files_name) * samples_per_game, 8*8),
                dtype=int
            )
            
            idx = 0
            for gm_idx, gm_name in tqdm(enumerate(self.game_files_name), 
                                        desc="Loading games"):
                
                game_log = load_game_log(self.path_dataset + gm_name)
                last_board_state = copy.copy(game_log[0][-1])
                is_black_winner = isBlackWinner(game_log[1][-1], last_board_state)
                
                # Pour chaque position dans le jeu
                for sm_idx in range(30):
                    
                    # Si use_all_samples=True, on prend les coups des deux joueurs
                    if self.use_all_samples:
                        # Coups du joueur noir (indices pairs)
                        self._add_sample(
                            idx, game_log, 2*sm_idx, is_black=True
                        )
                        idx += 1
                        
                        # Coups du joueur blanc (indices impairs)
                        self._add_sample(
                            idx, game_log, 2*sm_idx + 1, is_black=False
                        )
                        idx += 1
                    else:
                        # Ancien comportement: seulement les coups du gagnant
                        if is_black_winner:
                            end_move = 2*sm_idx
                            is_current_black = True
                        else:
                            end_move = 2*sm_idx + 1
                            is_current_black = False
                        
                        self._add_sample(idx, game_log, end_move, is_current_black)
                        idx += 1
        
        print(f"Number of samples: {len(self.samples)}")
        print(f"Use all samples (winners + losers): {self.use_all_samples}")
    
    def _add_sample(self, idx, game_log, end_move, is_black):
        """
        Ajoute un échantillon au dataset
        
        Args:
            idx: Index dans le dataset
            game_log: Log du jeu
            end_move: Index du coup final
            is_black: Si True, c'est un coup du joueur noir
        """
        # Extraire la séquence de features
        if end_move + 1 >= self.len_samples:
            features = game_log[0][end_move - self.len_samples + 1:end_move + 1]
        else:
            features = [self.starting_board_stat]
            # Padding avec l'état initial
            for i in range(self.len_samples - end_move - 2):
                features.append(self.starting_board_stat)
            # Ajouter les états du jeu
            for i in range(end_move + 1):
                features.append(game_log[0][i])
        
        # Si c'est le joueur noir, multiplier par -1
        if is_black:
            features = np.array([features], dtype=int) * -1
        else:
            features = np.array([features], dtype=int)
        
        self.samples[idx] = features
        self.outputs[idx] = np.array(game_log[1][end_move]).flatten()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.load_data_once4all:
            features = self.samples[idx]
            y = self.outputs[idx]
        else:
            # Version sans chargement en mémoire (à implémenter si nécessaire)
            raise NotImplementedError("Mode sans chargement complet non implémenté")
        
        return features, y, idx


def get_dataset_stats(dataset_conf):
    """
    Calcule les statistiques d'un dataset
    
    Args:
        dataset_conf: Configuration du dataset
        
    Returns:
        dict: Statistiques du dataset
    """
    # Compter le nombre de jeux
    with open(dataset_conf["filelist"]) as f:
        n_games = len([line for line in f])
    
    use_all = dataset_conf.get("use_all_samples", True)
    samples_per_game = 60 if use_all else 30
    total_samples = n_games * samples_per_game
    
    stats = {
        'n_games': n_games,
        'samples_per_game': samples_per_game,
        'total_samples': total_samples,
        'use_all_samples': use_all,
        'len_samples': dataset_conf['len_samples']
    }
    
    return stats


def create_balanced_dataset(dataset_conf, balance_ratio=1.0):
    """
    Crée un dataset équilibré entre gagnants et perdants
    
    Args:
        dataset_conf: Configuration du dataset
        balance_ratio: Ratio perdants/gagnants (1.0 = équilibré)
        
    Returns:
        CustomDatasetAll: Dataset équilibré
    """
    # Forcer l'utilisation de tous les samples
    dataset_conf["use_all_samples"] = True
    
    # Créer le dataset complet
    full_dataset = CustomDatasetAll(dataset_conf)
    
    # Si balance_ratio = 1.0, retourner le dataset complet
    if balance_ratio == 1.0:
        return full_dataset
    
    # Sinon, implémenter l'équilibrage (pour plus tard)
    # TODO: Sous-échantillonner ou sur-échantillonner selon le ratio
    
    return full_dataset


if __name__ == "__main__":
    # Test du module
    print("="*80)
    print("TEST DU MODULE DATA ÉTENDU")
    print("="*80)
    
    # Configuration de test
    dataset_conf = {
        "filelist": "train.txt",
        "len_samples": 5,
        "path_dataset": "./dataset/",
        "batch_size": 64,
        "use_all_samples": True  # Utiliser toutes les données
    }
    
    print("\n1. Test avec TOUTES les données (gagnants + perdants):")
    print("-" * 60)
    stats_all = get_dataset_stats(dataset_conf)
    print(f"Nombre de jeux: {stats_all['n_games']}")
    print(f"Échantillons par jeu: {stats_all['samples_per_game']}")
    print(f"Total échantillons: {stats_all['total_samples']}")
    print(f"Utilise tous les samples: {stats_all['use_all_samples']}")
    
    print("\n2. Test avec SEULEMENT les gagnants (ancien mode):")
    print("-" * 60)
    dataset_conf["use_all_samples"] = False
    stats_winners = get_dataset_stats(dataset_conf)
    print(f"Nombre de jeux: {stats_winners['n_games']}")
    print(f"Échantillons par jeu: {stats_winners['samples_per_game']}")
    print(f"Total échantillons: {stats_winners['total_samples']}")
    print(f"Utilise tous les samples: {stats_winners['use_all_samples']}")
    
    print("\n3. Comparaison:")
    print("-" * 60)
    ratio = stats_all['total_samples'] / stats_winners['total_samples']
    print(f"Ratio d'augmentation des données: {ratio}x")
    print(f"Échantillons supplémentaires: {stats_all['total_samples'] - stats_winners['total_samples']}")
    
    print("\n" + "="*80)
    print("✓ Tests terminés!")
    print("="*80)
