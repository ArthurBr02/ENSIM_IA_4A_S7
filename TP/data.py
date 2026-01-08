import h5py
import numpy as np
import torch
from tqdm import tqdm
import copy
from utile import has_tile_to_flip,isBlackWinner,initialze_board,BOARD_SIZE
from torch.utils.data import Dataset

# Fonctions pour la data augmentation
def rotate_board(board, k):
    """Rotate board by k*90 degrees counterclockwise."""
    return np.rot90(board, k)

def flip_board(board):
    """Flip board horizontally (symmetry)."""
    return np.fliplr(board)

def rotate_move(move, k):
    """Rotate move matrix by k*90 degrees counterclockwise."""
    move_matrix = move.reshape(8, 8)
    rotated = np.rot90(move_matrix, k)
    return rotated.flatten()

def flip_move(move):
    """Flip move matrix horizontally (symmetry)."""
    move_matrix = move.reshape(8, 8)
    flipped = np.fliplr(move_matrix)
    return flipped.flatten()

# Method to load the game log from an HDF5 file
def load_game_log(file_path):
    # file_path: path to the HDF5 file containing the game log
    h5f = h5py.File(file_path, 'r')  # Open the HDF5 file in read mode
    game_name = file_path.split('/')[-1].replace(".h5", "")  # Extract the game name from the file path
    game_log = np.array(h5f[game_name][:])  # Read the game log data as a NumPy array
    h5f.close()  # Close the HDF5 file
    return game_log  # Return the loaded game log


class SampleManager():
    def __init__(self,
                 game_name,
                 file_dir,
                 end_move,
                 len_moves,
                 isBlackPlayer):
        
        ''' each sample is a sequence of board states 
        from index (end_move - len_moves) to inedx end_move
        
        file_dir : directory of dataset
        game_name: name of file (game)
        end_move : the index of last recent move 
        len_moves: length of sequence
        isBlackPlayer: register the turn : True if it is a move of black player
        	(if black is the current player the board should be multiplay by -1)
        '''
        
        self.file_dir=file_dir
        self.game_name=game_name
        self.end_move=end_move
        self.len_moves=len_moves
        self.isBlackPlayer=isBlackPlayer
    
    def set_file_dir(self, file_dir):
        self.file_dir=file_dir
    def set_game_name(self, game_name):
        self.game_name=game_name
    def set_end_move(self, end_move):
        self.end_move=end_move
    def set_len_moves(self, len_moves):
        self.len_moves=len_moves

class CustomDatasetMany(Dataset):
    def __init__(self,
                 dataset_conf,load_data_once4all=True):
        """
        Custom dataset class for Othello game.

        Parameters:
        - dataset_conf (dict): Configuration dictionary containing dataset parameters.
        - load_data_once4all (bool): Flag indicating whether to load all data at once.
        """
                 
        self.load_data_once4all=load_data_once4all
        
        self.starting_board_stat=initialze_board()
        
        # self.filelist : a list of all games for train/dev/test
        self.filelist=dataset_conf["filelist"]
        #len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
        self.len_samples=dataset_conf["len_samples"] 
        self.path_dataset = dataset_conf["path_dataset"]
        
        #read all file name from train/dev/test.txt files
        with open(self.filelist) as f:
            list_files = [line.rstrip() for line in f]
        self.game_files_name=list_files#[s + ".h5" for s in list_files]       
        
        if self.load_data_once4all:
            # Utiliser float32 au lieu de int pour PyTorch
            self.samples=np.zeros((len(self.game_files_name)*30,self.len_samples,8,8), dtype=np.float32)
            self.outputs=np.zeros((len(self.game_files_name)*30,8*8), dtype=np.float32)
            idx=0
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):

                game_log=load_game_log(self.path_dataset+gm_name)
                last_board_state=copy.copy(game_log[0][-1])
                is_black_winner=isBlackWinner(game_log[1][-1],last_board_state)
                for sm_idx in range(30):
                    if is_black_winner:
                        end_move=2*sm_idx
                    else:
                        end_move=2*sm_idx+1
                        
                    if end_move+1 >= self.len_samples:
                        features=game_log[0][end_move-self.len_samples+1:
                                             end_move+1]
                    else:
                        features=[self.starting_board_stat]
                        #Padding starting board state before first index of sequence
                        for i in range(self.len_samples-end_move-2):
                            features.append(self.starting_board_stat)
                        #adding the inital of game as the end of sequence sample
                        for i in range(end_move+1):
                            features.append(game_log[0][i])

                    #if black is the current player the board should be multiplay by -1    
                    if is_black_winner:       
                        features=np.array([features],dtype=np.float32)*-1
                    else:
                        features=np.array([features],dtype=np.float32)    
                        
                    self.samples[idx]=features
                    self.outputs[idx]=np.array(game_log[1][end_move]).flatten()
                    idx+=1
        else:
        
            #creat a list of samples as SampleManager objcets
            self.samples=np.empty(len(self.game_files_name)*30, dtype=object)
            idx=0
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log=load_game_log(self.path_dataset+gm_name)
                last_board_state=copy.copy(game_log[0][-1])
                is_black_winner=isBlackWinner(game_log[1][-1],last_board_state)
                for sm_idx in range(30):
                    if is_black_winner:
                        end_move=2*sm_idx
                    else:
                        end_move=2*sm_idx+1
                    self.samples[idx]=SampleManager(gm_name,
                                                    self.path_dataset,
                                                    end_move,
                                                    self.len_samples,
                                                    is_black_winner)
                    idx+=1
        
        #np.random.shuffle(self.samples)
        print(f"Number of samples : {len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        
        if self.load_data_once4all:
            # Retourner directement les tensors sans conversion
            features = self.samples[idx]
            y = self.outputs[idx]
        else:
            game_log=load_game_log(self.samples[idx].file_dir+self.samples[idx].game_name)
            if self.samples[idx].end_move+1 >= self.samples[idx].len_moves:
                features=game_log[0][self.samples[idx].end_move-self.samples[idx].len_moves+1:
                                     self.samples[idx].end_move+1]
            else:
                features=[self.starting_board_stat]
                #Padding starting board state before first index of sequence
                for i in range(self.samples[idx].len_moves-self.samples[idx].end_move-2):
                    features.append(self.starting_board_stat)
                #adding the inital of game as the end of sequence sample
                for i in range(self.samples[idx].end_move+1):
                    features.append(game_log[0][i])

            #if black is the current player the board should be multiplay by -1    
            if self.samples[idx].isBlackPlayer:       
                features=np.array([features],dtype=float)*-1
            else:
                features=np.array([features],dtype=float)

            #y is a move matrix
            y=np.array(game_log[1][self.samples[idx].end_move]).flatten()
            
        return features,y,self.len_samples
    

class CustomDatasetOne(Dataset):
    def __init__(self,
                 dataset_conf,load_data_once4all=True):
        """
        Custom dataset class for Othello game.

        Parameters:
        - dataset_conf (dict): Configuration dictionary containing dataset parameters.
        - load_data_once4all (bool): Flag indicating whether to load all data at once.
        """
                 
        self.load_data_once4all=load_data_once4all
        
        self.starting_board_stat=initialze_board()
        
        # self.filelist : a list of all games for train/dev/test
        self.filelist=dataset_conf["filelist"]
        #len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
        self.len_samples=dataset_conf["len_samples"] 
        self.path_dataset = dataset_conf["path_dataset"]
        
        #read all file name from train/dev/test.txt files
        with open(self.filelist) as f:
            list_files = [line.rstrip() for line in f]
        self.game_files_name=list_files#[s + ".h5" for s in list_files]       
        
        if self.load_data_once4all:
            self.samples=np.zeros((len(self.game_files_name)*30,self.len_samples,8,8), dtype=int)
            self.outputs=np.zeros((len(self.game_files_name)*30,8*8), dtype=int)
            idx=0
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log=load_game_log(self.path_dataset+gm_name)
                last_board_state=copy.copy(game_log[0][-1])
                is_black_winner=isBlackWinner(game_log[1][-1],last_board_state)
                for sm_idx in range(30):
                    if is_black_winner:
                        end_move=2*sm_idx
                    else:
                        end_move=2*sm_idx+1
                        
                    if end_move+1 >= self.len_samples:
                        features=game_log[0][end_move-self.len_samples+1:
                                             end_move+1]
                    else:
                        features=[self.starting_board_stat]
                        #Padding starting board state before first index of sequence
                        for i in range(self.len_samples-end_move-2):
                            features.append(self.starting_board_stat)
                        #adding the inital of game as the end of sequence sample
                        for i in range(end_move+1):
                            features.append(game_log[0][i])

                    #if black is the current player the board should be multiplay by -1    
                    if is_black_winner:       
                        features=np.array([features],dtype=int)*-1
                    else:
                        features=np.array([features],dtype=int)    
                        
                    self.samples[idx]=features
                    self.outputs[idx]=np.array(game_log[1][end_move]).flatten()
                    idx+=1
        else:
        
            #creat a list of samples as SampleManager objcets
            self.samples=np.empty(len(self.game_files_name)*30, dtype=object)
            idx=0
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log=load_game_log(self.path_dataset+gm_name)
                last_board_state=copy.copy(game_log[0][-1])
                is_black_winner=isBlackWinner(game_log[1][-1],last_board_state)
                for sm_idx in range(30):
                    if is_black_winner:
                        end_move=2*sm_idx
                    else:
                        end_move=2*sm_idx+1
                    self.samples[idx]=SampleManager(gm_name,
                                                    self.path_dataset,
                                                    end_move,
                                                    self.len_samples,
                                                    is_black_winner)
                    idx+=1
        
        print(f"Number of samples : {len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        if self.load_data_once4all:
            features=self.samples[idx]
            y=self.outputs[idx]
        else:
            game_log=load_game_log(self.samples[idx].file_dir+self.samples[idx].game_name)

            if self.samples[idx].end_move+1 >= self.samples[idx].len_moves:
                features=game_log[0][self.samples[idx].end_move-self.samples[idx].len_moves+1:
                                     self.samples[idx].end_move+1]
            else:
                features=[self.starting_board_stat]
                #Padding starting board state before first index of sequence
                for i in range(self.samples[idx].len_moves-self.samples[idx].end_move-2):
                    features.append(self.starting_board_stat)
                #adding the inital of game as the end of sequence sample
                for i in range(self.samples[idx].end_move+1):
                    features.append(game_log[0][i])

            #if black is the current player the board should be multiplay by -1    
            if self.samples[idx].isBlackPlayer:       
                features=np.array([features],dtype=float)*-1
            else:
                features=np.array([features],dtype=float)

            #y is a move matrix
            y=np.array(game_log[1][self.samples[idx].end_move]).flatten()
            
        return features,y,self.len_samples

    



class CustomDatasetManyAugmented(Dataset):
    def __init__(self,
                 dataset_conf,load_data_once4all=True):
        """
        Custom dataset class for Othello game with data augmentation.
        Applies rotations (90°, 180°, 270°) and horizontal flips for each rotation.

        Parameters:
        - dataset_conf (dict): Configuration dictionary containing dataset parameters.
        - load_data_once4all (bool): Flag indicating whether to load all data at once.
        """
                 
        self.load_data_once4all=load_data_once4all
        
        self.starting_board_stat=initialze_board()
        
        # self.filelist : a list of all games for train/dev/test
        self.filelist=dataset_conf["filelist"]
        #len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
        self.len_samples=dataset_conf["len_samples"] 
        self.path_dataset = dataset_conf["path_dataset"]
        
        #read all file name from train/dev/test.txt files
        with open(self.filelist) as f:
            list_files = [line.rstrip() for line in f]
        self.game_files_name=list_files#[s + ".h5" for s in list_files]       
        
        if self.load_data_once4all:
            # Avec augmentation: 1 original + 3 rotations = 4, chacune avec flip = 4*2 = 8 versions
            num_augmentations = 8
            total_samples = len(self.game_files_name) * 30 * num_augmentations
            
            # Utiliser float32 au lieu de int pour PyTorch
            self.samples=np.zeros((total_samples, self.len_samples, 8, 8), dtype=np.float32)
            self.outputs=np.zeros((total_samples, 8*8), dtype=np.float32)
            idx=0
            
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log=load_game_log(self.path_dataset+gm_name)
                last_board_state=copy.copy(game_log[0][-1])
                is_black_winner=isBlackWinner(game_log[1][-1],last_board_state)
                
                for sm_idx in range(30):
                    if is_black_winner:
                        end_move=2*sm_idx
                    else:
                        end_move=2*sm_idx+1
                        
                    if end_move+1 >= self.len_samples:
                        features=game_log[0][end_move-self.len_samples+1:
                                             end_move+1]
                    else:
                        features=[self.starting_board_stat]
                        #Padding starting board state before first index of sequence
                        for i in range(self.len_samples-end_move-2):
                            features.append(self.starting_board_stat)
                        #adding the inital of game as the end of sequence sample
                        for i in range(end_move+1):
                            features.append(game_log[0][i])

                    #if black is the current player the board should be multiplay by -1    
                    if is_black_winner:       
                        features=np.array([features],dtype=np.float32)*-1
                    else:
                        features=np.array([features],dtype=np.float32)    
                    
                    original_output = np.array(game_log[1][end_move]).flatten()
                    
                    # Appliquer les rotations: 0° (original), 90°, 180°, 270°
                    # Note: k=1 -> 90°, k=2 -> 180°, k=3 -> 270°
                    for k in range(4):  # 0, 1, 2, 3 rotations de 90°
                        # Rotation des features (toutes les boards de la séquence)
                        rotated_features = np.array([rotate_board(board, k) for board in features[0]])
                        rotated_output = rotate_move(original_output, k)
                        
                        # Version avec rotation seulement
                        self.samples[idx] = rotated_features
                        self.outputs[idx] = rotated_output
                        idx += 1
                        
                        # Version avec rotation + symétrie (flip)
                        flipped_features = np.array([flip_board(board) for board in rotated_features])
                        flipped_output = flip_move(rotated_output)
                        
                        self.samples[idx] = flipped_features
                        self.outputs[idx] = flipped_output
                        idx += 1
        else:
            # Pour le mode lazy loading, on garde les indices et on applique l'augmentation à la volée
            base_samples = len(self.game_files_name) * 30
            num_augmentations = 8
            self.samples=np.empty(base_samples * num_augmentations, dtype=object)
            idx=0
            
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log=load_game_log(self.path_dataset+gm_name)
                last_board_state=copy.copy(game_log[0][-1])
                is_black_winner=isBlackWinner(game_log[1][-1],last_board_state)
                
                for sm_idx in range(30):
                    if is_black_winner:
                        end_move=2*sm_idx
                    else:
                        end_move=2*sm_idx+1
                    
                    # Créer 8 versions augmentées
                    for aug_idx in range(num_augmentations):
                        rotation_k = aug_idx // 2  # 0,0,1,1,2,2,3,3
                        apply_flip = aug_idx % 2 == 1  # False,True,False,True,...
                        
                        self.samples[idx]=SampleManager(gm_name,
                                                        self.path_dataset,
                                                        end_move,
                                                        self.len_samples,
                                                        is_black_winner)
                        # Stocker les paramètres d'augmentation
                        self.samples[idx].rotation_k = rotation_k
                        self.samples[idx].apply_flip = apply_flip
                        idx+=1
        
        print(f"Number of samples (with augmentation): {len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.load_data_once4all:
            # Retourner directement les tensors sans conversion
            features = self.samples[idx]
            y = self.outputs[idx]
        else:
            sample_manager = self.samples[idx]
            game_log=load_game_log(sample_manager.file_dir+sample_manager.game_name)
            
            if sample_manager.end_move+1 >= sample_manager.len_moves:
                features=game_log[0][sample_manager.end_move-sample_manager.len_moves+1:
                                     sample_manager.end_move+1]
            else:
                features=[self.starting_board_stat]
                #Padding starting board state before first index of sequence
                for i in range(sample_manager.len_moves-sample_manager.end_move-2):
                    features.append(self.starting_board_stat)
                #adding the inital of game as the end of sequence sample
                for i in range(sample_manager.end_move+1):
                    features.append(game_log[0][i])

            #if black is the current player the board should be multiplay by -1    
            if sample_manager.isBlackPlayer:       
                features=np.array([features],dtype=float)*-1
            else:
                features=np.array([features],dtype=float)

            #y is a move matrix
            y=np.array(game_log[1][sample_manager.end_move]).flatten()
            
            # Appliquer l'augmentation
            rotation_k = sample_manager.rotation_k
            apply_flip = sample_manager.apply_flip
            
            # Rotation
            features = np.array([rotate_board(board, rotation_k) for board in features[0]])
            y = rotate_move(y, rotation_k)
            
            # Flip si nécessaire
            if apply_flip:
                features = np.array([flip_board(board) for board in features])
                y = flip_move(y)
            
            features = np.array([features], dtype=float)
            
        return features,y,self.len_samples


class CustomDatasetOneAugmented(Dataset):
    def __init__(self,
                 dataset_conf,load_data_once4all=True):
        """
        Custom dataset class for Othello game with data augmentation.
        Applies rotations (90°, 180°, 270°) and horizontal flips for each rotation.

        Parameters:
        - dataset_conf (dict): Configuration dictionary containing dataset parameters.
        - load_data_once4all (bool): Flag indicating whether to load all data at once.
        """
                 
        self.load_data_once4all=load_data_once4all
        
        self.starting_board_stat=initialze_board()
        
        # self.filelist : a list of all games for train/dev/test
        self.filelist=dataset_conf["filelist"]
        #len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
        self.len_samples=dataset_conf["len_samples"] 
        self.path_dataset = dataset_conf["path_dataset"]
        
        #read all file name from train/dev/test.txt files
        with open(self.filelist) as f:
            list_files = [line.rstrip() for line in f]
        self.game_files_name=list_files#[s + ".h5" for s in list_files]       
        
        if self.load_data_once4all:
            # Avec augmentation: 1 original + 3 rotations = 4, chacune avec flip = 4*2 = 8 versions
            num_augmentations = 8
            total_samples = len(self.game_files_name) * 30 * num_augmentations
            
            self.samples=np.zeros((total_samples, self.len_samples, 8, 8), dtype=int)
            self.outputs=np.zeros((total_samples, 8*8), dtype=int)
            idx=0
            
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log=load_game_log(self.path_dataset+gm_name)
                last_board_state=copy.copy(game_log[0][-1])
                is_black_winner=isBlackWinner(game_log[1][-1],last_board_state)
                
                for sm_idx in range(30):
                    if is_black_winner:
                        end_move=2*sm_idx
                    else:
                        end_move=2*sm_idx+1
                        
                    if end_move+1 >= self.len_samples:
                        features=game_log[0][end_move-self.len_samples+1:
                                             end_move+1]
                    else:
                        features=[self.starting_board_stat]
                        #Padding starting board state before first index of sequence
                        for i in range(self.len_samples-end_move-2):
                            features.append(self.starting_board_stat)
                        #adding the inital of game as the end of sequence sample
                        for i in range(end_move+1):
                            features.append(game_log[0][i])

                    #if black is the current player the board should be multiplay by -1    
                    if is_black_winner:       
                        features=np.array([features],dtype=int)*-1
                    else:
                        features=np.array([features],dtype=int)    
                    
                    original_output = np.array(game_log[1][end_move]).flatten()
                    
                    # Appliquer les rotations: 0° (original), 90°, 180°, 270°
                    # Note: k=1 -> 90°, k=2 -> 180°, k=3 -> 270°
                    for k in range(4):  # 0, 1, 2, 3 rotations de 90°
                        # Rotation des features (toutes les boards de la séquence)
                        rotated_features = np.array([rotate_board(board, k) for board in features[0]])
                        rotated_output = rotate_move(original_output, k)
                        
                        # Version avec rotation seulement
                        self.samples[idx] = rotated_features
                        self.outputs[idx] = rotated_output
                        idx += 1
                        
                        # Version avec rotation + symétrie (flip)
                        flipped_features = np.array([flip_board(board) for board in rotated_features])
                        flipped_output = flip_move(rotated_output)
                        
                        self.samples[idx] = flipped_features
                        self.outputs[idx] = flipped_output
                        idx += 1
        else:
            # Pour le mode lazy loading, on garde les indices et on applique l'augmentation à la volée
            base_samples = len(self.game_files_name) * 30
            num_augmentations = 8
            self.samples=np.empty(base_samples * num_augmentations, dtype=object)
            idx=0
            
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log=load_game_log(self.path_dataset+gm_name)
                last_board_state=copy.copy(game_log[0][-1])
                is_black_winner=isBlackWinner(game_log[1][-1],last_board_state)
                
                for sm_idx in range(30):
                    if is_black_winner:
                        end_move=2*sm_idx
                    else:
                        end_move=2*sm_idx+1
                    
                    # Créer 8 versions augmentées
                    for aug_idx in range(num_augmentations):
                        rotation_k = aug_idx // 2  # 0,0,1,1,2,2,3,3
                        apply_flip = aug_idx % 2 == 1  # False,True,False,True,...
                        
                        self.samples[idx]=SampleManager(gm_name,
                                                        self.path_dataset,
                                                        end_move,
                                                        self.len_samples,
                                                        is_black_winner)
                        # Stocker les paramètres d'augmentation
                        self.samples[idx].rotation_k = rotation_k
                        self.samples[idx].apply_flip = apply_flip
                        idx+=1
        
        print(f"Number of samples (with augmentation): {len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.load_data_once4all:
            features=self.samples[idx]
            y=self.outputs[idx]
        else:
            sample_manager = self.samples[idx]
            game_log=load_game_log(sample_manager.file_dir+sample_manager.game_name)

            if sample_manager.end_move+1 >= sample_manager.len_moves:
                features=game_log[0][sample_manager.end_move-sample_manager.len_moves+1:
                                     sample_manager.end_move+1]
            else:
                features=[self.starting_board_stat]
                #Padding starting board state before first index of sequence
                for i in range(sample_manager.len_moves-sample_manager.end_move-2):
                    features.append(self.starting_board_stat)
                #adding the inital of game as the end of sequence sample
                for i in range(sample_manager.end_move+1):
                    features.append(game_log[0][i])

            #if black is the current player the board should be multiplay by -1    
            if sample_manager.isBlackPlayer:       
                features=np.array([features],dtype=float)*-1
            else:
                features=np.array([features],dtype=float)

            #y is a move matrix
            y=np.array(game_log[1][sample_manager.end_move]).flatten()
            
            # Appliquer l'augmentation
            rotation_k = sample_manager.rotation_k
            apply_flip = sample_manager.apply_flip
            
            # Rotation
            features = np.array([rotate_board(board, rotation_k) for board in features[0]])
            y = rotate_move(y, rotation_k)
            
            # Flip si nécessaire
            if apply_flip:
                features = np.array([flip_board(board) for board in features])
                y = flip_move(y)
            
            features = np.array([features], dtype=float)
            
        return features,y,self.len_samples

