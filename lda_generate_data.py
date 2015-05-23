from pandas.core.frame import DataFrame

import numpy as np
import pylab as plt


class LdaDataGenerator:
    
        def __init__(self, alpha):
            self.alpha = alpha

        def generate_word_dists(self, n_topics, vocab_size, document_length):
            
            width = vocab_size/n_topics
            word_dists = np.zeros((n_topics, vocab_size))
     
            for k in range(n_topics):
                temp = np.zeros((n_topics, width))
                temp[k, :] = int(document_length / width)
                word_dists[k,:] = temp.flatten()
     
            word_dists /= word_dists.sum(axis=1)[:, np.newaxis] # turn counts into probabilities     
#             self._plot_nicely(word_dists, 'Topics X Terms', 'Terms', 'Topics')
            return word_dists              
        
        def generate_document(self, word_dists, n_topics, vocab_size, document_length):

            # sample topic proportions with uniform dirichlet parameter alpha of length n_topics
            theta = np.random.mtrand.dirichlet([self.alpha] * n_topics)

            # for every word in the vocab for this document
            d = np.zeros(vocab_size)
            for n in range(document_length):
            
                # sample a new topic index    
                k = np.random.multinomial(1, theta).argmax()
                
                # sample a new word from the word distribution of topic k
                w = np.random.multinomial(1, word_dists[k,:]).argmax()

                # increase the occurrence of word w in document d
                d[w] += 1

            return d
        
        def generate_input_df(self, n_topics, vocab_size, document_length, n_docs):
                        
            # word_dists is the topic x document_length matrix
            word_dists = self.generate_word_dists(n_topics, vocab_size, document_length)            
            
            # generate each document x terms vector
            docs = np.zeros((vocab_size, n_docs))
            for i in range(n_docs):
                docs[:, i] = self.generate_document(word_dists, n_topics, vocab_size, document_length)
                
            df = DataFrame(docs)
            df = df.transpose()
            print df.shape            
#             self._plot_nicely(df, 'Documents X Terms', 'Terms', 'Docs')
            return df
        
        def _plot_nicely(self, mat, title, xlabel, ylabel):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.matshow(mat)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect(2)
            ax.set_aspect('auto')
            plt.colorbar(im)
            # plt.savefig(filename)
            plt.show()