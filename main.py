#####################
# Load the packages that is necessary to train and test the neural networks
# in the code, Pytorch and RDkit is maily utilized.
# To handle the graph neural network, Torch Geometric package has been used
####################

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
import torch
import rdkit
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import networkx as nx
from rdkit.Chem import MolFromSmiles
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import sys, os
import time
from torch_geometric.data import DataLoader 
from sklearn.model_selection import train_test_split 

# To minimize the random fluctuation, random seed has been fixed

import random
import numpy as np
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.cuda.manual_seed_all(2020) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(2020)
random.seed(2020)

#class for preprocessing of the data from dataset
class Prep:
    def __init__(self,dataset_file):
        self.data = dataset_file
#################################
# mol atom feature for mol graph
#################################
    def atom_features(self,atom):
    
        return np.array(self.one_of_k_encoding_unk(atom.GetSymbol(),
                                              ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                               'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                               'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                               'Pt', 'Hg', 'Pb', 'X']) +
                        self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        [atom.GetIsAromatic()])
    
    
    # one ont encoding
    def one_of_k_encoding(self,x, allowable_set):
        if x not in allowable_set:
            raise Exception(
                'input {0} not in allowable set{1}:'.format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))
    
    
    
    def one_of_k_encoding_unk(self,x, allowable_set):
        '''Maps inputs not in the allowable set to the last element.'''
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))
    

#######################################
# SMILES to molecular graph edge index
#######################################


    def smile_to_graph(self,smile):
        mol = Chem.MolFromSmiles(smile)
        c_size = mol.GetNumAtoms()
        features = []
        for atom in mol.GetAtoms():
            feature = self.atom_features(atom)
            features.append(feature / sum(feature))
    
        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        mol_adj = np.zeros((c_size, c_size))
        for e1, e2 in g.edges:
            mol_adj[e1, e2] = 1
    
        mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
        index_row, index_col = np.where(mol_adj >= 0.5)
        for i, j in zip(index_row, index_col):
            edge_index.append([i, j])
    
        return c_size, features, edge_index
    
    
    def create_dataset_for_train(self,dataset, fold=5):
        # load dataset
        data = pd.read_csv(self.data)
        solubility = data.pop('class')  
        X_train, X_val, solubility_train, solubility_val = train_test_split(
            data, solubility, test_size=(1/fold), random_state=2020)
    
        solvent_train_ = X_train.pop('solvent')
        solute_train_ = X_train.pop('solute')
        solvent_val_ = X_val.pop('solvent')
        solute_val_ = X_val.pop('solute')
        solubility_val = solubility_val.tolist()
        solubility_train = solubility_train.tolist()
        
        solvent_train = []
        solvent_val = []
        solute_train = []
        solute_val = []
    
        for smile in solvent_train_:
            solvent_train.append(smile)
        for smile in solute_train_:
            solute_train.append(smile)
        for smile in solvent_val_:
            solvent_val.append(smile)
        for smile in solute_val_:
            solute_val.append(smile)
    
    #################################################
    # removing errors of SMILES
    ##################################################
        error1 = []
        for i, v in enumerate(solute_val):
            mol = Chem.MolFromSmiles(v)
            if mol == None:
                error1.append(i)
            else:
                pass
        error1.reverse()  
        for n in error1:
            solvent_val.pop(n)
            solute_val.pop(n)
            solubility_val.pop(n)
    
        error2 = []
        for i, v in enumerate(solute_train):
            mol = Chem.MolFromSmiles(v)
            if mol == None:
                error2.append(i)
            else:
                pass
        error2.reverse()  
        for n in error2:
            solvent_train.pop(n)
            solute_train.pop(n)
            solubility_train.pop(n)
            
        # convert smiles to canonical smiles
        # create solvent_train smiles graphs
        smile_graph_solvent_train = []
        solvent_train_canonical = []
        for smile in solvent_train:
            solvent_train_canonical.append(Chem.MolToSmiles(Chem.MolFromSmiles(smile)))
        for smile in solvent_train_canonical:
            g = self.smile_to_graph(smile)
            smile_graph_solvent_train.append(g)
        # convert smiles to canonical smiles
        # create solute_train smiles graphs
        smile_graph_solute_train = []
        solute_train_canonical = []
        for smile in solute_train:
            solute_train_canonical.append(Chem.MolToSmiles(Chem.MolFromSmiles(smile)))    
        for smile in solute_train_canonical:
            g = self.smile_to_graph(smile)
            smile_graph_solute_train.append(g)
            
        # convert smiles to canonical smiles
        # create solvent_val smiles graphs
        smile_graph_solvent_val = []
        solvent_val_canonical = []
        for smile in solvent_val:
            solvent_val_canonical.append(Chem.MolToSmiles(Chem.MolFromSmiles(smile)))
        for smile in solvent_val_canonical:
            g = self.smile_to_graph(smile)
            smile_graph_solvent_val.append(g)
        # convert smiles to canonical smiles
        # create solute_train smiles graphs
        smile_graph_solute_val = []
        solute_val_canonical = []
        for smile in solute_val:
            solute_val_canonical.append(Chem.MolToSmiles(Chem.MolFromSmiles(smile)))
        for smile in solute_val_canonical:
            g = self.smile_to_graph(smile)
            smile_graph_solute_val.append(g)
    
    
        train_dataset = SolubilityDataset(root='data', dataset=dataset,
                                          solute_smiles=solute_train,
                                          solvent_smiles=solvent_train,
                                          y=solubility_train,
                                          smile_graph_solute=smile_graph_solute_train,
                                          smile_graph_solvent=smile_graph_solvent_train)
    
        valid_dataset = SolubilityDataset(root='data', dataset=dataset,
                                        solute_smiles=solute_val,
                                        solvent_smiles=solvent_val,
                                        y=solubility_val,
                                        smile_graph_solute=smile_graph_solute_val,
                                        smile_graph_solvent=smile_graph_solvent_val)
    
        return train_dataset, valid_dataset
    




#############
# utils
#############

# initialize the dataset
class SolubilityDataset(InMemoryDataset,Prep):
    def __init__(self, root='/tmp', dataset=None,
                 y=None, transform=None, pre_transform=None,
                 solute_smiles=None, solvent_smiles=None,
                 smile_graph_solute=None, smile_graph_solvent=None):

        super(SolubilityDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(solute_smiles, solvent_smiles, y, smile_graph_solute, smile_graph_solvent)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_solute.pt', self.dataset + '_data_solvent.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, solute_smiles, solvent_smiles, y, smile_graph_solute, smile_graph_solvent):
        assert (len(solute_smiles) == len(smile_graph_solvent) and len(solute_smiles) == len(y))
        # The three lists must be the same length!
        # if the lists has different length, there would be a error during removing errors of SMILES or critical errors of the dataset.
        
        data_list_solute = []
        data_list_solvent = []
        data_len = len(y)
        for i in range(data_len):
            solute_smile = solute_smiles[i]
            solvent_smile = solvent_smiles[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size_solute, features_solute, edge_index_solute = self.smile_to_graph(solute_smile)
            c_size_solvent, features_solvent, edge_index_solvent = self.smile_to_graph(solvent_smile)

            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData_solute = DATA.Data(x=torch.Tensor(features_solute),
                                       edge_index=torch.LongTensor(
                                           edge_index_solute).transpose(1, 0),
                                       y=torch.FloatTensor([labels]))
            GCNData_solute.__setitem__('c_size_solute', torch.LongTensor([c_size_solute]))

            GCNData_solvent = DATA.Data(x=torch.Tensor(features_solvent),
                                        edge_index=torch.LongTensor(
                                            edge_index_solvent).transpose(1, 0),
                                        y=torch.FloatTensor([labels]))
            GCNData_solvent.__setitem__('c_size_solvent', torch.LongTensor([c_size_solvent]))

            data_list_solute.append(GCNData_solute)
            data_list_solvent.append(GCNData_solvent)

        if self.pre_filter is not None:
            data_list_solute = [data for data in data_list_solute if self.pre_filter(data)]
            data_list_solvent = [data for data in data_list_solvent if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_solute = [self.pre_transform(data) for data in data_list_solute]
            data_list_solvent = [self.pre_transform(data) for data in data_list_solvent]
        self.data_solute = data_list_solute
        self.data_solvent = data_list_solvent

    def __len__(self):
        return len(self.data_solute)

    def __getitem__(self, idx):
        return self.data_solute[idx], self.data_solvent[idx]





# training function at each epoch



def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 10
    TRAIN_BATCH_SIZE = 64
    loss_fn = torch.nn.MSELoss()
    for batch_idx, data in enumerate(train_loader):
        data_solute = data[0].to(device)
        data_solvent = data[1].to(device)
        optimizer.zero_grad()
        output = model(data_solute, data_solvent)
        loss = loss_fn(output, data_solute.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

# predict


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_solute = data[0].to(device)
            data_solvent = data[1].to(device)
            output = model(data_solute, data_solvent)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_solute.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


# prepare the solute and solvent pairs
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB




class GNNNet(torch.nn.Module):
    def __init__(self,
                 n_output=1,
                 num_features_solvent=78,
                 num_features_solute=78,
                 output_dim=128,
                 dropout=0.3):
        super(GNNNet, self).__init__()

        print('GNNNet Loaded')
        self.n_output = n_output

        self.solute_conv1 = GCNConv(num_features_solute,
                                    num_features_solute * 2)
        self.solute_conv2 = GCNConv(num_features_solute * 2,
                                    num_features_solute * 4)
        self.solute_conv3 = GCNConv(num_features_solute * 4,
                                    num_features_solute * 4)
        self.solute_fc_g1 = torch.nn.Linear(num_features_solute * 4, 1024)
        self.solute_fc_g2 = torch.nn.Linear(1024, 1024)
        self.solute_fc_g3 = torch.nn.Linear(1024, 1024)
        self.solute_fc_g4 = torch.nn.Linear(1024, output_dim)

        self.solvent_conv1 = GCNConv(num_features_solvent,
                                     num_features_solvent * 2)
        self.solvent_conv2 = GCNConv(num_features_solvent * 2,
                                     num_features_solvent * 4)
        self.solvent_conv3 = GCNConv(num_features_solvent * 4,
                                     num_features_solvent * 4)
        self.solvent_fc_g1 = torch.nn.Linear(num_features_solvent * 4, 1024)
        self.solvent_fc_g2 = torch.nn.Linear(1024, 1024)
        self.solvent_fc_g3 = torch.nn.Linear(1024, 1024)
        self.solvent_fc_g4 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_solute, data_solvent):
        # get solute input
        solute_x, solute_edge_index, solute_batch = data_solute.x, data_solute.edge_index, data_solute.batch
        # get solvent input
        solvent_x, solvent_edge_index, solvent_batch = data_solvent.x, data_solvent.edge_index, data_solvent.batch

        x = self.solute_conv1(solute_x, solute_edge_index)
        x = self.relu(x)
        x = self.solute_conv2(x, solute_edge_index)
        x = self.relu(x)
        x = self.solute_conv3(x, solute_edge_index)
        x = self.relu(x)
        x = gep(x, solute_batch)  # global pooling
        # flatten
        x = self.relu(self.solute_fc_g1(x))
        x = self.dropout(x)
        x = self.solute_fc_g2(x)
        x = self.dropout(x)

        x = self.solute_fc_g4(x)
        x = self.dropout(x)

        xt = self.solvent_conv1(solvent_x, solvent_edge_index)
        xt = self.relu(xt)
        xt = self.solvent_conv2(xt, solvent_edge_index)
        xt = self.relu(xt)
        xt = self.solvent_conv3(xt, solvent_edge_index)
        xt = self.relu(xt)
        xt = gep(xt, solvent_batch)  # global pooling

        # flatten
        xt = self.relu(self.solvent_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.solvent_fc_g2(xt)
        xt = self.dropout(xt)
        xt = self.solvent_fc_g4(xt)
        xt = self.dropout(xt)
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)

        xc = self.dropout(xc)
        out = self.out(xc)
        return out


#loading the prepreocessing class 
prep = Prep("solubility dataset.csv")

start = time.time()  
     
datasets = ['train']
cuda_name = "cuda"
print('cuda_name:', cuda_name)
fold = 5
cross_validation_flag = True


TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = 0.0001
NUM_EPOCHS = 1000
DECAY = 1e-5
print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
print('\n')

models_dir = 'models'
results_dir = 'results'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Main program: iterate over different datasets
result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
model = GNNNet()
model.to(device)
model_st = GNNNet.__name__
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)
for dataset in datasets:
    train_data, valid_data = prep.create_dataset_for_train(dataset, fold)
   
    train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=TRAIN_BATCH_SIZE,
                                                shuffle=True,
                                                collate_fn=collate)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                                batch_size=TEST_BATCH_SIZE,
                                                shuffle=False,
                                                collate_fn=collate)

    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1
    model_file_name = 'models/model_' + model_st + '_' + dataset + '_' + str(
        fold) + '.model'

    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)
        print('predicting for valid data')
        G, P = predicting(model, device, valid_loader)
        modP = []
        for i in P:
            roundP = float(round(i))
            modP.append(roundP)
        val = get_mse(G, modP)  #calculating the mse

        print('valid result:', val, best_mse)
        if val < best_mse:
            best_mse = val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            print('RMSE improved at epoch ', best_epoch, '; best_test_mse',
                  best_mse, model_st, dataset, fold, '-fold')
            print("Current Running Time : {} sec".format(time.time() - start))
            if epoch == 0:
                saved_time = time.time()
                print('\n')
            if epoch == NUM_EPOCHS - 1:
                print('\n')

            if epoch != 0:
                print("Estimated Remaining Time: {} min".format(
                    ((time.time() - saved_time) / epoch * NUM_EPOCHS + saved_time - time.time()) / 60))
                print('\n')

        else:
            print('No improvement since epoch ', best_epoch, '; best_test_mse',
                  best_mse, model_st, dataset, fold, '-fold')
            print("Current running time : {} sec".format(time.time() - start))
            print("Estimated Remaining Time: {} min".format(
                ((time.time() - saved_time) / epoch * NUM_EPOCHS + saved_time - time.time()) / 60))
            print('\n')

print("\n\n\n")
print("Total Working Time : {} sec".format(time.time() -
                                            start))  





print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
'''
saving the trained model.
'''
torch.save(model.state_dict(), '\models')    




def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_solute = data[0].to(device)
            data_solvent = data[1].to(device)
            # data = data.to(device)
            output = model(data_solute, data_solvent)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat(
                (total_labels, data_solute.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def load_model(model_path):
    model = torch.load(model_path)
    return model

if __name__ == '__main__':
    dataset = 'test'  # dataset selection
    model_st = GNNNet.__name__
    print('dataset:', dataset)

    cuda_name = "cuda"
    print('cuda_name:', cuda_name)

    TEST_BATCH_SIZE = 16
    models_dir = 'models'
    results_dir = 'results'

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    #     model_file_name = 'models/model_' + model_st + '_' + dataset + '_5.model'
    result_file_name = 'results/result_' + model_st + '_' + dataset + '.txt'

    model = GNNNet()
    model.to(device)
    model.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
    test_data = prep.create_dataset_for_test(dataset)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=TEST_BATCH_SIZE,
                                              shuffle=False,
                                              collate_fn=collate)

    Y, P = predicting(model, device, test_loader)

    print('Prediction finished')





modP = []
for i in P:
    roundP = float(round(i))
    modP.append(roundP)
    
answer = []
for i in range(len(modP)):
    if modP[i] == Y[i]:
        answer.append(1)
    else:
        answer.append(0)


#for showing the 
# print(answer)
# accuracy_score(Y, modP)

# A, B = predicting(model, device, train_loader)
# modB = []
# for i in B:
#     roundB = float(round(i))
#     modB.append(roundB)

# print('Accuracy of Train set:', accuracy_score(A, modB))
# print('Accuracy of Test set:', accuracy_score(Y, modP))
# print('ROC_AUC of Train set:', roc_auc_score(A, modB))
# print('ROC_AUC of Test set:', roc_auc_score(Y, modP))




        
    
        
        
    