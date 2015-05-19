"""
Cross-validation for LDA
"""

import sys

from lda_for_fragments import Ms2Lda
from lda_cgs import CollapseGibbsLda
from lda_generate_data import LdaDataGenerator
import numpy as np
import pandas as pd
import pylab as plt


class CrossValidatorLda:
    
    def __init__(self, df, K, alpha, beta):
        self.df = df
        self.K = K
        self.alpha = alpha
        self.beta = beta

    def cross_validate(self, n_folds, n_burn, n_samples, n_thin):
    
        shuffled_df = self.df.reindex(np.random.permutation(self.df.index))
        folds = np.array_split(shuffled_df, n_folds)
                
        print "K=" + str(self.K)
        testing_hms = []
        for i in range(len(folds)):
            
            training_df = None
            testing_df = None
            testing_idx = -1
            for j in range(len(folds)):
                if j == i:
                    print "Testing fold=" + str(j)
                    testing_df = folds[j]
                    testing_idx = j
                else:
                    print "Training fold=" + str(j)
                    if training_df is None:
                        training_df = folds[j]
                    else:
                        training_df.append(folds[j])

            print "Run training gibbs"
            training_gibbs = CollapseGibbsLda(training_df, self.K, self.alpha, self.beta)
            training_gibbs.run(n_burn, n_samples, n_thin)
            
            print "Run testing gibbs"
            training_ckn = training_gibbs.ckn
            training_ck = training_gibbs.ck
            testing_gibbs = CollapseGibbsLda(testing_df, self.K, self.alpha, self.beta, training_ckn, training_ck)
            testing_gibbs.run(n_burn, n_samples, n_thin)
        
            # testing_hm = stats.hmean(testing_gibbs.all_lls)
            testing_hm = len(testing_gibbs.all_lls) / np.sum(1.0/testing_gibbs.all_lls) 
            print "Harmonic mean for testing fold " + str(testing_idx) + " = " + str(testing_hm)
            print
            testing_hms.append(testing_hm)
            
        testing_hm = np.array(testing_hms)
        mean_marg = np.mean(testing_hm)
        print "Cross-validation done!"
        print "K=" + str(self.K) + ", mean_approximate_log_marginal_likelihood=" + str(mean_marg)
    
def main():

#     K = 10
#     print "MS2LDA K=" + str(K)
#     alpha = 0.1
#     beta = 0.01    
#     n_docs = 50
#     vocab_size = 200
#     document_length = 50
#     gen = LdaDataGenerator(alpha)
#     df = gen.generate_input_df(K, vocab_size, document_length, n_docs)
# 
#     cv = CrossValidatorLda(df, K, alpha, beta)
#     cv.cross_validate(n_folds=4, n_burn=100, n_samples=200, n_thin=10)    

    K = int(sys.argv[1])
    print "Cross-validation for K=" + str(K)
    n_folds = 5
    n_samples = 400
    n_burn = 200
    n_thin = 10
    alpha = 0.1
    beta = 0.01
    
    fragment_filename = 'input/Beer_3_T10_POS_fragments.csv'
    neutral_loss_filename = 'input/Beer_3_T10_POS_losses.csv'
    mzdiff_filename = None    
    ms1_filename = 'input/Beer_3_T10_POS_ms1.csv'
    ms2_filename = 'input/Beer_3_T10_POS_ms2.csv'
    ms2lda = Ms2Lda(fragment_filename, neutral_loss_filename, mzdiff_filename, 
                ms1_filename, ms2_filename)
    
    df = ms2lda.preprocess()
    cv = CrossValidatorLda(df, K, alpha, beta)
    cv.cross_validate(n_folds, n_burn, n_samples, n_thin)    
        
if __name__ == "__main__":
    main()