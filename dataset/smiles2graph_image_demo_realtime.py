from rdkit import Chem
import numpy as np
import time
import pickle
from rdkit.Chem.rdchem import BondType, BondDir, ChiralType
import os
import datetime
from rdkit.Chem import Draw


BOND_TYPE = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}
BOND_DIR = {BondDir.NONE: 0, BondDir.ENDUPRIGHT: 1, BondDir.ENDDOWNRIGHT: 2}
CHI = {ChiralType.CHI_UNSPECIFIED: 0, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_TETRAHEDRAL_CCW: 2, ChiralType.CHI_OTHER: 3}

def bond_dir(bond):
    d = bond.GetBondDir()
    return BOND_DIR[d]

def bond_type(bond):
    t = bond.GetBondType()
    return BOND_TYPE[t]

def atom_chiral(atom):
    c = atom.GetChiralTag()
    return CHI[c]

def atom_to_feature(atom):
    num = atom.GetAtomicNum() - 1
    if num == -1:
        # atom.GetAtomicNum() is 0, which is the generic wildcard atom *, may be used to symbolize an unknown atom of any element.
        # See https://biocyc.org/help.html?object=smiles
        num = 118  # normal num is [0, 117], so we use 118 to denote wildcard atom *
    return [num, atom_chiral(atom)]

def bond_to_feature(bond):
    return [bond_type(bond), bond_dir(bond)]

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 2
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph 


def Smiles2Img(smis, size=224, savePath=None):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png
    '''
    # try:
    mol = Chem.MolFromSmiles(smis)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
    if savePath is not None:
        img.save(savePath)
    return img
    # except:
    #     return None


def convert_smiles(smi):
    try:
        g = smiles2graph(smi)
        img = Smiles2Img(smi)
        return g, img
    except:
        return None, None
