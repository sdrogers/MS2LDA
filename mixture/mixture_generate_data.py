""" Produces some synthetic data according to the generative process of mixture model.
See example usage in the main method of mixture_cgs.py
"""
from numpy import int64
from pandas.core.frame import DataFrame

import numpy as np
import pylab as plt
import pandas as pd
from ms2lda.lda_generate_data import LdaDataGenerator


class MixtureDataGenerator(LdaDataGenerator):

        def __init__(self, alpha, make_plot=False):
            LdaDataGenerator.__init__(self, alpha, make_plot)            
            self.last_k = -1
                
        def generate_document(self, word_dists, n_cluster, vocab_size, document_length):

            # sample topic proportions with uniform dirichlet parameter alpha of length n_cluster
            theta = np.random.mtrand.dirichlet([self.alpha] * n_cluster)

            # sample a new cluster index for the whole document
            k = np.random.multinomial(1, theta).argmax()

#             # instead assign each document to a new cluster            
#             k = self.last_k + 1
#             self.last_k = k

            # for every word in the vocab for this document
            d = np.zeros(vocab_size)
            for n in range(document_length):
                            
                # sample a new word from the word distribution of cluster k
                w = np.random.multinomial(1, word_dists[k,:]).argmax()

                # increase the occurrence of word w in document d
                d[w] += 1

            return d
