"""
An implementation of a collapsed Gibbs sampling for Latent Dirichlet Allocation [1]
but with 3-bags of words for each word type (fragment, loss, mzdiff)

[1] Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." 
the Journal of machine Learning research 3 (2003): 993-1022.

"""

import cPickle
import sys
import time

from numpy import int32
from numpy.random import RandomState
from scipy.special import psi

from justin.lda_generate_data import LdaDataGenerator
import justin.lda_utils as utils
from lda_3bags_cgs_numba import sample_numba  
from lda_3bags_cgs_numpy import sample_numpy
import numpy as np
import pylab as plt
from lda_3bags_model import bag_of_word_dtype, bags

class CollapseGibbs_3bags_Lda(object):
    
    def __init__(self, df, vocab, K, alpha, beta, random_state=None, previous_model=None):
        """
        Initialises the collapsed Gibbs sampling for LDA
        
        Arguments:
        - df: the dataframe of counts of vocabularies x documents
        - K: no. of topics
        - alpha: symmetric prior on document-topic assignment
        - beta: symmetric prior on word-topic assignment
        - previous_model: previous LDA run, if any
        """
        
        print "CGS LDA initialising"
        self.df = df.replace(np.nan, 0)
        self.alpha = alpha            

        self.D = df.shape[0]    # total no of docs
        self.N = df.shape[1]    # total no of words
        assert(len(vocab)==self.N)
        
        # index of each word
        self.vocab = [v[0] for v in vocab] 
        
        # the bag index for each word in self.vocab, values can only be in [0, 1, ..., n_bags-1]
        self.vocab_type = [int(v[1]) for v in vocab]
            
        # assume exactly 3 bags
        self.bag_labels = bags
        self.n_bags = len(self.bag_labels)

        if hasattr(beta, "__len__"):
            # beta is an np array, must be the same length as the number of bags
            assert(len(beta)==self.n_bags)
            self.beta = beta
        else:
            # beta is a scalar, convert it into np array
            self.beta = np.array([beta])
            self.beta = np.repeat(self.beta, self.n_bags)

        # set total no of topics
        self.cv = False
        self.previous_model = previous_model
        if self.previous_model is not None:
            
            # if some old topics were fixed
            if hasattr(self.previous_model, 'selected_topics'):
            
                # no. of new topics
                self.K = K
            
                # no. of previously selected topics
                self.previous_K = len(self.previous_model.selected_topics)                
                
                # Get the previous ckn and ck values from the training stage.
                # During gibbs update in this testing stage, assignment of word 
                # to the first previous_K topics will use the previous fixed 
                # topic-word distributions -- as specified by previous_ckn and previous_ck
                self.previous_ckn = self.previous_model.selected_ckn
                self.previous_ck = self.previous_model.selected_ck
                self.previous_vocab = self.previous_model.selected_vocab
                assert(len(self.previous_ck)==self.previous_K)
                assert(self.previous_ckn.shape[0]==len(self.previous_ck))
                assert(self.previous_ckn.shape[1]==len(self.previous_vocab))
                
                # make previous_ckn have the right number of columns
                N_diff = self.N - len(self.previous_vocab)
                temp = np.zeros((self.previous_K, N_diff), dtype=bag_of_word_dtype)
                self.previous_ckn = np.hstack((self.previous_ckn, temp)) # size is previous_K x N
                
                # make previous_ckn have the right number of rows
                temp = np.zeros((self.K, self.N), dtype=bag_of_word_dtype)
                self.previous_ckn = np.vstack((self.previous_ckn, temp)) # size is (previous_K+K) x N

                # make previous_ck have the right length
                temp = np.zeros(self.K, dtype=bag_of_word_dtype)
                self.previous_ck = np.hstack((self.previous_ck, temp)) # length is (previous_K+K)

                # total no. of topics = old + new topics
                self.K = self.K + self.previous_K
                print "Total no. of topics = " + str(self.K)
                                
                
            else:                
                raise ValueError("No previous topics have been selected")
                
        else:

            # for training stage
            self.K = K            
            self.previous_ckn = np.zeros((self.K, self.N), dtype=bag_of_word_dtype)
            self.previous_ck = np.zeros(self.K, dtype=bag_of_word_dtype)        
            self.previous_K = 0 # no old topics

        # make the current arrays too
        self.ckn = np.zeros((self.K, self.N), dtype=bag_of_word_dtype)
        self.ck = np.zeros(self.K, dtype=bag_of_word_dtype)        
        self.cdk = np.zeros((self.D, self.K), int32)
        self.cd = np.zeros(self.D, int32)

        # make sure to get the same results from running gibbs each time
        if random_state is None:
            self.random_state = RandomState(1234567890)
        else:
            self.random_state = random_state

        # randomly assign words to topics
        self.Z = {}        
        for d in range(self.D):
            if d%10==0:
                sys.stdout.write('.')
                sys.stdout.flush()
            document = self.df.iloc[[d]]
            word_idx = utils.word_indices(document)
            for pos, n in enumerate(word_idx):
                b = self.vocab_type[n]
                k = self.random_state.randint(self.K)
                self.cdk[d, k] += 1
                self.cd[d] += 1
                bag_label = self.bag_labels[b]
                bag_ckn = self.ckn[bag_label]
                bag_ck = self.ck[bag_label]
                bag_ckn[k, n] += 1
                bag_ck[k] += 1
                self.Z[(d, pos)] = k
        print

        # turn word counts in the document into a vector of word occurences
        self.document_indices = {}
        for d in range(self.D):
            document = self.df.iloc[[d]]
            word_idx = utils.word_indices(document)
            word_locs = []
            for pos, n in enumerate(word_idx):
                word_locs.append((pos, n))
            self.document_indices[d] = word_locs
            
    def get_posterior_alpha(self, n_iter=100):
        """
        Estimate posterior alpha of the Dirichlet-Multinomial for doc-topic using the last sample
        see Minka, T. P. (2003). Estimating a Dirichlet distribution. Annals of Physics, 2000(8), 1-13. http://doi.org/10.1007/s00256-007-0299-1
        """
         
        # initialise old and new alphas before iteration
        alpha_old = np.ones(self.K) * self.alpha
        alpha_new = np.zeros_like(alpha_old)
        for i in range(n_iter):
            numerator = 0
            denominator = 0
            for d in range(self.D):
                numerator += psi(self.cdk[d] + alpha_old) - psi(alpha_old)
                denominator += psi(np.sum(self.cdk[d] + alpha_old)) - psi(np.sum(alpha_old))
            alpha_new = alpha_old * (numerator/denominator)
            
            # set alpha_new to alpha_old for the next iteration update
            alpha_old = alpha_new  

        return alpha_new            
                                                    
    def run(self, n_burn, n_samples, n_thin, use_native=False):
        """ 
        Runs the Gibbs sampling for LDA 
        
        Arguments:
        - n_burn: no of initial burn-in samples
        - n_samples: no of samples, must be > n_burn
        - n_thin: how often to thin the log_likelihood results stored
        - use_native: if True, will call the sampling function in lda_cgs_numba.py
        """

        # select the sampler function to use
        sampler_func = None
        if not use_native:
            print "Using Numpy for q-bags LDA sampling"
            sampler_func = sample_numpy
        else:
            # not quite sure how to write a generic numba version for any number of bags ...
            if (self.n_bags<=3):
                print "Using Numba for 3-bags LDA sampling"
                sampler_func = sample_numba                
            else:
                # fallback to the numpy version for now
                print "FALLBACK to using Numpy for q-bags LDA sampling"                
                sampler_func = sample_numpy

        # this will modify the various count matrices (Z, cdk, ckn, cd, ck) inside
        self.topic_word_, self.doc_topic_, self.loglikelihoods_ = sampler_func(
                self.random_state, n_burn, n_samples, n_thin,
                self.D, self.N, self.K, self.document_indices,
                self.alpha, self.beta,
                self.Z, self.cdk, self.cd, self.previous_K,
                self.ckn, self.ck, self.previous_ckn, self.previous_ck,
                self.vocab_type, self.bag_labels)     
        
        # update posterior alpha from the last sample  
        self.posterior_alpha = self.get_posterior_alpha()           
        
    @classmethod
    def load(cls, filename):
        f = file(filename, 'rb')
        obj = cPickle.load(f)
        f.close()
        print "Model loaded from " + filename
        return obj
    
    def save(self, topic_indices, model_out, words_out):
        
        self.selected_topics = topic_indices
        self.selected_ckn = self.ckn[topic_indices, :].copy()
        self.selected_ck = self.ck[topic_indices].copy()        

        # Accumulate the positions of the words with zero counts across all bags
        all_bags_zero_pos = set()
        for bi in range(len(self.bag_labels)):

            bag_label = self.bag_labels[bi]
            bag_ckn = self.selected_ckn[bag_label]
            colsum = np.sum(bag_ckn, axis=0)

            # store the positions of words with zero counts for removal 
            all_zero_cols = (colsum==0)
            all_zero_cols_pos = np.where(all_zero_cols)[0]
            if len(all_bags_zero_pos) == 0:
                # if empty then just add everything
                all_bags_zero_pos.update(all_zero_cols_pos)
            else:
                # otherwise keep the intersection only
                all_bags_zero_pos.intersection_update(all_zero_cols_pos)
            
        total_words = self.selected_ckn.shape[1]
        all_bags_non_zero_pos = set(range(total_words)) - all_bags_zero_pos
                
        all_bags_zero_pos = sorted(list(all_bags_zero_pos))
        all_bags_non_zero_pos = sorted(list(all_bags_non_zero_pos))

        # delete words with all zeros counts from the matrix
        self.selected_ckn = np.delete(self.selected_ckn, all_bags_zero_pos, 1)
        
        # save the words with non-zero counts in the matrix
        self.selected_vocab = []
        for pos in all_bags_non_zero_pos:
            word = self.vocab[pos]
            word_type = self.vocab_type[pos]
            tup = (word, word_type)
            self.selected_vocab.append(tup)                
        self.selected_vocab = np.array(self.selected_vocab)

        # dump the whole model out
        # binary mode ('b') is required for portability between Unix and Windows
        f = file(model_out, 'wb')
        cPickle.dump(self, f)
        f.close()
        print "Model saved to " + model_out

        # also write out the selected vocabs into a text file
        # can be used for feature processing later ..
        with open(words_out, 'w') as f:
            for item in self.selected_vocab:
                f.write("{}\n".format(item))                
        print "Words written to " + words_out            
                                    
def main():

    multiplier = 1
    n_topics = 20 * multiplier
    n_docs = 100 * multiplier
    vocab_size = 200 * multiplier
    document_length = 50 * multiplier

    alpha = 0.1
    beta = 0.01    
    n_samples = 200
    n_burn = 0
    n_thin = 1
    
    random_state = RandomState(1234567890)

    gen = LdaDataGenerator(alpha, make_plot=True)
    df, vocab = gen.generate_input_df(n_topics, vocab_size, document_length, n_docs, 
                                      previous_vocab=None, vocab_prefix='gibbs1', 
                                      df_outfile='input/test1.csv', vocab_outfile='input/test1.words', n_bags=3)
    df, vocab = gen.generate_from_file('input/test1.csv', 'input/test1.words')

    print "\nUsing own LDA"
    gibbs1 = CollapseGibbs_3bags_Lda(df, vocab, n_topics, alpha, beta, random_state=random_state, previous_model=None)
    start_time = time.time()
    gibbs1.run(n_burn, n_samples, n_thin, use_native=True)
    print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))
    print gibbs1.posterior_alpha    
      
#     # try saving model
#     selected_topics = [0, 1, 2, 3, 4, 5]
#     gibbs1.save(selected_topics, 'input/gibbs1.p', 'input/gibbs1.selected.words')
#     
#     # try loading model
#     gibbs1 = CollapseGibbs_3bags_Lda.load('input/gibbs1.p')
#     if hasattr(gibbs1, 'selected_topics'):
#         print "Kept topics = " + str(gibbs1.selected_topics)
#    
#     gen._plot_nicely(gibbs1.doc_topic_.T, 'Inferred Topics X Docs', 'docs', 'topics', outfile='test1_doc_topic.png')
#     for b in range(gibbs1.n_bags):
#         gen._plot_nicely(gibbs1.topic_word_[b], 'Inferred Topics X Terms for bag ' + str(b), 'terms', 'topics', outfile='test1_topic_word.png')
#     plt.plot(gibbs1.loglikelihoods_)
#     plt.show()    
#     
#     EPSILON = 0.05
#     n_top_words = 20    
#     print_topic_words(gibbs1.topic_word_, gibbs1.n_bags, n_top_words, n_topics, vocab, EPSILON)
#             
#     # now run gibbs again on another df with the few selected topics above
#     gen = LdaDataGenerator(alpha, make_plot=True)
#     df2, vocab2 = gen.generate_input_df(n_topics, vocab_size, document_length, n_docs, 
#                                         previous_vocab=gibbs1.selected_vocab, vocab_prefix='gibbs2', 
#                                         df_outfile='input/test2.csv', vocab_outfile='input/test2.words', n_bags=3)
#     df2, vocab2 = gen.generate_from_file('input/test2.csv', 'input/test2.words') 
#      
#     gibbs2 = CollapseGibbs_3bags_Lda(df2, vocab2, n_topics, alpha, beta, previous_model=gibbs1)
#     gibbs2.run(n_burn, n_samples, n_thin, use_native=True)
#      
#     gen._plot_nicely(gibbs2.doc_topic_.T, 'Inferred Topics X Docs', 'docs', 'topics', outfile='test2_doc_topic.png')
#     for b in range(gibbs1.n_bags):
#         gen._plot_nicely(gibbs2.topic_word_[b], 'Inferred Topics X Terms for bag ' + str(b), 'terms', 'topics', 
#                          outfile='test2_topic_word.png')
#     plt.plot(gibbs2.loglikelihoods_)
#     plt.show()
#     print_topic_words(gibbs2.topic_word_, gibbs2.n_bags, n_top_words, n_topics, vocab2, EPSILON)
        
def print_topic_words(topic_words, n_bags, n_top_words, n_topics, vocab, EPSILON=0.05):

    words = [item[0] for item in vocab]
    words_bag = [item[1] for item in vocab]
    
    # for all topics
    for k in range(n_topics):

        print('Topic {}'.format(k))
        
        # for each bag in the topic
        for b in range(n_bags):
            
            # get topic-word distribution of the bag
            topic_dists = topic_words[b]
            topic_dist = topic_dists[k]
            
            # print top words in each bag
            sorted_idx = np.argsort(topic_dist)[:-n_top_words:-1]
            print('\tbag {}: '.format(b)),
            for j in sorted_idx:
                w = words[j]
                word_bag = words_bag[j]
                val = topic_dist[j]
                if val > EPSILON:
                    print (w + ' (word_bag ' + str(word_bag) + ', ' + str(val) + ') '), 
            print    

if __name__ == "__main__":
    main()