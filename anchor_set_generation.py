import os
import torch
from torch.nn.functional import normalize
import copy
import numpy as np

from stlenc.phis_generator import StlGenerator
from stlenc.traj_measure import BaseMeasure
from stlenc.orig_utils import from_string_to_formula, load_pickle, dump_pickle
from stlenc.kernel import StlKernel


def anchorGeneration(diff_init = False, # to control whether we want formulae to be semantically different by construction
                     embed_dim: int = 30, # embedding dimension, aka number of generated formulae in the anchor set
                     n_vars: int = 3, # dimension of the input signal (3D in this case)
                     leaf_prob: float = 0.4, # complexity of the generated formula
                     cosine_similarity_threshold: float = 0.8 # if two formulae cosine similarity exceeds 0.9, then discard one of the two
                    ) -> str:
    
    # initialize STL formula generator
    sampler = StlGenerator(leaf_prob)
    
    # effective anchor set generation
    if diff_init:
        
        # initialize the anchor set with a randomly sampled formula
        diff_anchor_set = [sampler.sample(nvars=n_vars)]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mu = BaseMeasure(device=device)

        # generates a set of random signals working as a tester for the formulae testing
        signals = mu.sample(samples=10000, varn=n_vars)

        # computes robustness value for the initial set of formulae in the anchor set
        anchor_rob_vectors = torch.cat([phi.quantitative(signals, normalize=True).unsqueeze(0) for phi in diff_anchor_set], 0)

        while len(diff_anchor_set) < embed_dim:
            # sample the 'remaining' formulae to reach the desired number of `embed_dim` formulae:
            candidate_anchors = sampler.bag_sample(embed_dim - len(diff_anchor_set), nvars = n_vars)
    
            # compute robustness of candidate anchor formulae on the same signals as previous anchor set
            candidate_robs = torch.cat([phi.quantitative(signals, normalize=True).unsqueeze(0) for phi in candidate_anchors], 0)
            
            # compute cosine similarity between current anchor set and candidate new formulae
            cos_simil = torch.tril(normalize(candidate_robs) @ normalize(anchor_rob_vectors).t(), diagonal=-1)

            # check which formulae are similar (i.e. greater cosine similarity then threshold) w.r.t. current anchors
            # NOTA: chiedere a gaia se cosine similarities negative vanno ammazzate con un valore assoluto o meno!
            similar_idx = [torch.where(cos_simil[r, :] > cosine_similarity_threshold)[0].tolist() for r in range(cos_simil.shape[0])]
    
            # keep only those who are semantically distant
            keep_idx = list(set(np.arange(len(candidate_anchors)).tolist()).difference(set([i for sublist in similar_idx for i in sublist])))
            
            diff_anchor_set += [copy.deepcopy(candidate_anchors[i]) for i in keep_idx]
            
            # Convert keep_idx to a tensor on the same device as candidate_robs
            keep_idx_tensor = torch.tensor(keep_idx, device=candidate_robs.device)
            
            # Use index_select to pick the relevant rows
            selected_robs = torch.index_select(candidate_robs, 0, keep_idx_tensor)
            
            # Concatenate on the same device
            anchor_rob_vectors = torch.cat([anchor_rob_vectors, copy.deepcopy(selected_robs)], dim=0)

            anchor_set = diff_anchor_set[:embed_dim]
            
    else:
        anchor_set = sampler.bag_sample(bag_size=embed_dim, nvars=n_vars) 

    filename = f'anchor_set_no_diff_{embed_dim}_dim'
    dump_pickle(filename, anchor_set)
    return filename


# EXAMPLE OF USAGE
anchorGeneration(diff_init = False, embed_dim = 1024)



