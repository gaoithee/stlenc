import os
import torch
from torch.nn.functional import normalize
import copy
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from stlenc.phis_generator import StlGenerator
from stlenc.traj_measure import BaseMeasure
from stlenc.orig_utils import from_string_to_formula, load_pickle, dump_pickle
from stlenc.kernel import StlKernel
from sklearn.decomposition import KernelPCA
from stlenc.anchor_set_generation import anchorGeneration
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class STLEncoder():
    def __init__(self, 
                 embed_dim: int,
                 anchor_filename: Optional[str] = None,
                 n_vars: int = 3,
                 pca = None):
        
        self.n_vars = n_vars
        self.embed_dim = embed_dim
        self.anchorset_filename = anchor_filename
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mu = BaseMeasure(device=self.device)
        self.kernel = StlKernel(self.mu, varn=self.n_vars)
        self.pca = pca

        if anchor_filename is None: 
            anchor_filename = anchorGeneration(diff_init = True, embed_dim = self.embed_dim, n_vars = self.n_vars)  
            anchor_filename+='.pickle'

        anchor_set = load_pickle(anchor_filename)
        self.anchor_set = anchor_set

    def compute_embeddings(self, formula: List[str], gram_matrix: None):
        if self.pca is None:
            return self.kernel.compute_bag_bag(formula, self.anchor_set)
        else:
            if gram_matrix is None:
                gram_matrix = self.kernel.compute_bag_bag(self.anchor_set, self.anchor_set)
                if hasattr(gram_matrix, 'numpy'):
                    gram_matrix = gram_matrix.numpy()
            else:
                gram_matrix = load_pickle('gram_matrix_1024.pickle')
            scaler = StandardScaler()
            scaled_gram = scaler.fit_transform(gram_matrix)
            pca = PCA(n_components=self.pca)
            pca.fit(scaled_gram)
            eigenvectors = pca.components_
            eigenvalues = pca.explained_variance_
            explained_variance = pca.explained_variance_ratio_
            gram_test = self.kernel.compute_bag_bag(formula, self.anchor_set)
            if hasattr(gram_test, 'numpy'):
                gram_test = gram_test.numpy()
            gram_test_scaled = scaler.transform(gram_test)
            transformed_test_data = pca.transform(gram_test_scaled)
            return transformed_test_data, eigenvectors, eigenvalues, explained_variance

# EXAMPLE OF USAGE
# df = pd.read_csv('datasets/test_balanced_validation_set.csv')
# formulae_to_embed = df['Formula']

# here we do not pass an anchor set so the Encoder creates a new one of dimension set to `embed_dim` 
# encoder = STLEncoder(embed_dim=1024, anchor_filename='anchor_set_1024_dim.pickle')
# formulae_embeddings = encoder.compute_embeddings(formulae_to_embed)
# print(formulae_embeddings)

