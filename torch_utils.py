# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:26:27 2023

@author: Frank
"""

# import packages
# general tools
import numpy as np
# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
# Pytorch and Pytorch Geometric
import torch
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_add_pool
# streamlit
import streamlit as st
import pandas as pd
import base64

class GNNModel(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.pool = global_add_pool
        self.lin1 = Linear(hidden_size, hidden_size)
        self.lin2 = Linear(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.pool(x, batch)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x.squeeze(dim=1)
     
def load_model():
    # Model must be created with again with with parameters
    gnn_model = GNNModel(num_features=79, hidden_size=32, num_classes=1)
    gnn_model.load_state_dict(torch.load('gnn_model.pth', map_location=torch.device('cpu')))
    gnn_model.eval()
    return gnn_model

model = load_model()
    
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)

def bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)

def molecular_graph(smiles):
    """
    Inputs:
    SMILES
    
    Outputs:
    Molecular graph
    
    """
        
    # convert SMILES to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)
    # get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2*mol.GetNumBonds()
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
    # construct node feature matrix X of shape (n_nodes, n_node_features)
    X = np.zeros((n_nodes, n_node_features))
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = atom_features(atom)
            
    X = torch.tensor(X, dtype = torch.float)
        
    # construct edge index array E of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    E = torch.stack([torch_rows, torch_cols], dim = 0)
        
    # construct edge feature array EF of shape (n_edges, n_edge_features)
    EF = np.zeros((n_edges, n_edge_features))
        
    for (k, (i,j)) in enumerate(zip(rows, cols)):
            
        EF[k] = bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
    EF = torch.tensor(EF, dtype = torch.float)
        
        # construct label tensor
        #y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
        
    # construct Pytorch Geometric data object and append to data list
    pgd = Data(x = X, edge_index = E, edge_attr = EF)
    
    return pgd



def get_predictions():
    # Collect input option
    input_option = st.radio("Select Input Option", ("SMILES", "CSV"))

    if input_option == "SMILES":
        # Collect SMILES inputs
        smiles = st.text_area("Enter the SMILES (one per line)")

        # Split the input into individual SMILES
        smiles_list = smiles.strip().split('\n')
    else:
        # Collect CSV file
        csv_file = st.file_uploader("Upload CSV file", type=["csv"])
        if csv_file is not None:
            # Read CSV file into a DataFrame
            df = pd.read_csv(csv_file)
            # Extract the SMILES column
            smiles_list = df['SMILES'].tolist()
        else:
            st.warning("Please upload a CSV file.")

    ok = st.button('Predict')

    if ok:
        if len(smiles_list) == 0:
            st.warning("No SMILES input found.")
            return

        # Create a DataFrame to store the results
        results = pd.DataFrame(columns=['SMILES', 'Prediction (K)'])

        # Iterate over each SMILES and make predictions
        for sm in smiles_list:
            # Convert SMILES to PyTorch Geometric graph data object
            graph_data = molecular_graph(sm)

            # Make forward pass to obtain predicted label
            with torch.no_grad():
                y_pred = model(graph_data)

            # Convert predicted label to a NumPy array and extract scalar value
            prediction = y_pred.cpu().numpy()[0]

            # Round the prediction to 1 decimal place
            #rounded_prediction = round(prediction, 1)

            # Append the SMILES and rounded prediction to the results DataFrame
            results = results.append({'SMILES': sm, 'Prediction (K)': f"{prediction:.2f}"}, ignore_index=True)

        # Display the results in a table
        st.table(results)

        # Add an option to download the results as a CSV file
        csv = results.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

