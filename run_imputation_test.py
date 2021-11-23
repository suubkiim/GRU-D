import torch
import argparse
import pandas as pd
import numpy as np
import torch.utils.data as utils

from tqdm import trange
from GRUD import *

parser = argparse.ArgumentParser(description='Missing Value imputation using GRU-D')
parser.add_argument('--data_pth', type=str, default='data/speed_matrix_2015.pkl')
parser.add_argument('--ckpt_pth', type=str, default='gru_d_0.pth')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def PrepareDataset(speed_matrix, \
                   BATCH_SIZE = 40, \
                   seq_len = 10, \
                   pred_len = 1,
                   masking = False, \
                   mask_ones_proportion = 0.8):
    """ Prepare training and testing datasets and dataloaders.
    
    Convert speed/volume/occupancy matrix to training and testing dataset. 
    The vertical axis of speed_matrix is the time axis and the horizontal axis 
    is the spatial axis.
    
    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    time_len = speed_matrix.shape[0]
    
    speed_matrix = speed_matrix.clip(0, 100)
    
    max_speed = speed_matrix.max().max()
    speed_matrix =  speed_matrix / max_speed
    
    speed_sequences, speed_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        speed_sequences.append(speed_matrix.iloc[i:i+seq_len].values)
        speed_labels.append(speed_matrix.iloc[i+seq_len:i+seq_len+pred_len].values)
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)
    
    # using zero-one mask to randomly set elements to zeros
    if masking:
        print('Split Speed finished. Start to generate Mask, Delta, Last_observed_X ...')
        np.random.seed(1024)
        Mask = np.random.choice([0,1], size=(speed_sequences.shape), p = [1 - mask_ones_proportion, mask_ones_proportion])
        speed_sequences = np.multiply(speed_sequences, Mask)
        
        # temporal information
        interval = 5 # 5 minutes
        S = np.zeros_like(speed_sequences) # time stamps
        for i in range(S.shape[1]):
            S[:,i,:] = interval * i

        Delta = np.zeros_like(speed_sequences) # time intervals
        for i in range(1, S.shape[1]):
            Delta[:,i,:] = S[:,i,:] - S[:,i-1,:]

        missing_index = np.where(Mask == 0)

        X_last_obsv = np.copy(speed_sequences)
        for idx in trange(missing_index[0].shape[0]):
            i = missing_index[0][idx] 
            j = missing_index[1][idx]
            k = missing_index[2][idx]
            if j != 0 and j != 9:
                Delta[i,j+1,k] = Delta[i,j+1,k] + Delta[i,j,k]
            if j != 0:
                X_last_obsv[i,j,k] = X_last_obsv[i,j-1,k] # last observation
        Delta = Delta / Delta.max() # normalize
    
    # shuffle and split the dataset to training and testing datasets
    print('Generate Mask, Delta, Last_observed_X finished. Start to shuffle and split dataset ...')
    sample_size = speed_sequences.shape[0]
    index = np.arange(sample_size, dtype = int)
    np.random.seed(1024)
    np.random.shuffle(index)
    
    speed_sequences = speed_sequences[index]
    speed_labels = speed_labels[index]
    
    if masking:
        X_last_obsv = X_last_obsv[index]
        Mask = Mask[index]
        Delta = Delta[index]
        speed_sequences = np.expand_dims(speed_sequences, axis=1)
        X_last_obsv = np.expand_dims(X_last_obsv, axis=1)
        Mask = np.expand_dims(Mask, axis=1)
        Delta = np.expand_dims(Delta, axis=1)
        dataset_agger = np.concatenate((speed_sequences, X_last_obsv, Mask, Delta), axis = 1)
            
    if masking:
        data, label = dataset_agger, speed_labels
    else:
        data, label = speed_sequences, speed_labels
    
    data, label = torch.Tensor(data), torch.Tensor(label)

    dataset = utils.TensorDataset(data, label)
    
    dataloader = utils.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)

    X_mean = np.mean(speed_sequences, axis = 0)
    
    print('Finished')
    
    return dataloader, max_speed, X_mean

# test_dataloader = torch.load('test_dataloader.pth')
# max_speed = torch.load('max_speed.pth')
# X_mean = torch.load('X_mean.pth')

speed_matrix =  pd.read_pickle(args.data_pth)

dataloader, max_speed, X_mean = PrepareDataset(speed_matrix, BATCH_SIZE = 64, masking = True)

# test_item = iter(test_dataloader).next()[0].to(device)

test_item = iter(dataloader).next()[0].to(device)

[_, type_size, step_size, fea_size] = test_item.size()
input_dim = fea_size
hidden_dim = fea_size
output_dim = fea_size

grud = GRUD(input_dim, hidden_dim, output_dim, X_mean, output_last = True).to(device)

grud.load_state_dict(torch.load(args.ckpt_pth))
grud.eval()

_, imputed = grud(test_item.to('cuda'))

print(">> input with missing values")
print(test_item)
print(">> input with imputation")
print(imputed)