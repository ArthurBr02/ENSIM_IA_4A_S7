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
            # Build lists dynamically to support variable-length games and use all moves
            samples_list = []
            outputs_list = []
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log = load_game_log(self.path_dataset+gm_name)
                num_moves = len(game_log[1])
                for end_move in range(num_moves):
                    if end_move+1 >= self.len_samples:
                        features = game_log[0][end_move-self.len_samples+1:end_move+1]
                    else:
                        features = [self.starting_board_stat]
                        # Padding starting board state before first index of sequence
                        for i in range(self.len_samples-end_move-2):
                            features.append(self.starting_board_stat)
                        # adding the initial of game as the end of sequence sample
                        for i in range(end_move+1):
                            features.append(game_log[0][i])

                    # if black is the current player the board should be multiplied by -1
                    # Determine which player plays at end_move: True if black
                    is_black_player = (end_move % 2 == 0)
                    features = np.array([features], dtype=np.float32)
                    if is_black_player:
                        features = features * -1

                    samples_list.append(features[0])
                    outputs_list.append(np.array(game_log[1][end_move]).flatten())

            # convert lists to arrays
            if len(samples_list) > 0:
                self.samples = np.array(samples_list, dtype=np.float32)
                self.outputs = np.array(outputs_list, dtype=np.float32)
            else:
                self.samples = np.zeros((0, self.len_samples, 8, 8), dtype=np.float32)
                self.outputs = np.zeros((0, 8*8), dtype=np.float32)
        else:
            # create a list of samples as SampleManager objects for lazy loading
            samples_list = []
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log = load_game_log(self.path_dataset+gm_name)
                num_moves = len(game_log[1])
                for end_move in range(num_moves):
                    # determine which player plays at end_move: True if black
                    is_black_player = (end_move % 2 == 0)
                    samples_list.append(SampleManager(gm_name,
                                                      self.path_dataset,
                                                      end_move,
                                                      self.len_samples,
                                                      is_black_player))

            self.samples = samples_list
        
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
            samples_list = []
            outputs_list = []
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log = load_game_log(self.path_dataset+gm_name)
                num_moves = len(game_log[1])
                for end_move in range(num_moves):
                    if end_move+1 >= self.len_samples:
                        features = game_log[0][end_move-self.len_samples+1:end_move+1]
                    else:
                        features = [self.starting_board_stat]
                        for i in range(self.len_samples-end_move-2):
                            features.append(self.starting_board_stat)
                        for i in range(end_move+1):
                            features.append(game_log[0][i])

                    is_black_player = (end_move % 2 == 0)
                    features = np.array([features], dtype=int)
                    if is_black_player:
                        features = features * -1

                    samples_list.append(features[0])
                    outputs_list.append(np.array(game_log[1][end_move]).flatten())

            if len(samples_list) > 0:
                self.samples = np.array(samples_list, dtype=int)
                self.outputs = np.array(outputs_list, dtype=int)
            else:
                self.samples = np.zeros((0, self.len_samples, 8, 8), dtype=int)
                self.outputs = np.zeros((0, 8*8), dtype=int)
        else:
            samples_list = []
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log = load_game_log(self.path_dataset+gm_name)
                num_moves = len(game_log[1])
                for end_move in range(num_moves):
                    is_black_player = (end_move % 2 == 0)
                    samples_list.append(SampleManager(gm_name,
                                                      self.path_dataset,
                                                      end_move,
                                                      self.len_samples,
                                                      is_black_player))

            self.samples = samples_list
        
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
            # Build augmented samples dynamically for all moves
            samples_list = []
            outputs_list = []
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log = load_game_log(self.path_dataset+gm_name)
                num_moves = len(game_log[1])
                for end_move in range(num_moves):
                    if end_move+1 >= self.len_samples:
                        features = game_log[0][end_move-self.len_samples+1:end_move+1]
                    else:
                        features = [self.starting_board_stat]
                        for i in range(self.len_samples-end_move-2):
                            features.append(self.starting_board_stat)
                        for i in range(end_move+1):
                            features.append(game_log[0][i])

                    is_black_player = (end_move % 2 == 0)
                    features = np.array([features], dtype=np.float32)
                    if is_black_player:
                        features = features * -1

                    original_output = np.array(game_log[1][end_move]).flatten()

                    for k in range(4):
                        rotated_features = np.array([rotate_board(board, k) for board in features[0]])
                        rotated_output = rotate_move(original_output, k)
                        samples_list.append(rotated_features)
                        outputs_list.append(rotated_output)

                        flipped_features = np.array([flip_board(board) for board in rotated_features])
                        flipped_output = flip_move(rotated_output)
                        samples_list.append(flipped_features)
                        outputs_list.append(flipped_output)

            if len(samples_list) > 0:
                self.samples = np.array(samples_list, dtype=np.float32)
                self.outputs = np.array(outputs_list, dtype=np.float32)
            else:
                self.samples = np.zeros((0, self.len_samples, 8, 8), dtype=np.float32)
                self.outputs = np.zeros((0, 8*8), dtype=np.float32)
        else:
            # Lazy mode: store SampleManager objects and augmentation params
            samples_list = []
            num_augmentations = 8
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log = load_game_log(self.path_dataset+gm_name)
                num_moves = len(game_log[1])
                for end_move in range(num_moves):
                    for aug_idx in range(num_augmentations):
                        rotation_k = aug_idx // 2
                        apply_flip = aug_idx % 2 == 1
                        sm = SampleManager(gm_name,
                                           self.path_dataset,
                                           end_move,
                                           self.len_samples,
                                           (end_move % 2 == 0))
                        sm.rotation_k = rotation_k
                        sm.apply_flip = apply_flip
                        samples_list.append(sm)

            self.samples = samples_list
        
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
            samples_list = []
            outputs_list = []
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log = load_game_log(self.path_dataset+gm_name)
                num_moves = len(game_log[1])
                for end_move in range(num_moves):
                    if end_move+1 >= self.len_samples:
                        features = game_log[0][end_move-self.len_samples+1:end_move+1]
                    else:
                        features = [self.starting_board_stat]
                        for i in range(self.len_samples-end_move-2):
                            features.append(self.starting_board_stat)
                        for i in range(end_move+1):
                            features.append(game_log[0][i])

                    is_black_player = (end_move % 2 == 0)
                    features = np.array([features], dtype=int)
                    if is_black_player:
                        features = features * -1

                    original_output = np.array(game_log[1][end_move]).flatten()
                    for k in range(4):
                        rotated_features = np.array([rotate_board(board, k) for board in features[0]])
                        rotated_output = rotate_move(original_output, k)
                        samples_list.append(rotated_features)
                        outputs_list.append(rotated_output)

                        flipped_features = np.array([flip_board(board) for board in rotated_features])
                        flipped_output = flip_move(rotated_output)
                        samples_list.append(flipped_features)
                        outputs_list.append(flipped_output)

            if len(samples_list) > 0:
                self.samples = np.array(samples_list, dtype=int)
                self.outputs = np.array(outputs_list, dtype=int)
            else:
                self.samples = np.zeros((0, self.len_samples, 8, 8), dtype=int)
                self.outputs = np.zeros((0, 8*8), dtype=int)
        else:
            samples_list = []
            num_augmentations = 8
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                game_log = load_game_log(self.path_dataset+gm_name)
                num_moves = len(game_log[1])
                for end_move in range(num_moves):
                    for aug_idx in range(num_augmentations):
                        rotation_k = aug_idx // 2
                        apply_flip = aug_idx % 2 == 1
                        sm = SampleManager(gm_name,
                                           self.path_dataset,
                                           end_move,
                                           self.len_samples,
                                           (end_move % 2 == 0))
                        sm.rotation_k = rotation_k
                        sm.apply_flip = apply_flip
                        samples_list.append(sm)

            self.samples = samples_list
        
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

