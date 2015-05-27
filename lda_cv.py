"""
Cross-validation for LDA
"""

import multiprocessing
import sys

import os

from joblib import Parallel, delayed  

from lda_cgs import CollapseGibbsLda
from lda_for_fragments import Ms2Lda
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
                
        testing_hms = []
        for i in range(len(folds)):
            
            training_df = None
            testing_df = None
            testing_idx = -1
            for j in range(len(folds)):
                if j == i:
                    print "K=" + str(self.K) + " Testing fold=" + str(j)
                    testing_df = folds[j]
                    testing_idx = j
                else:
                    print "K=" + str(self.K) + " Training fold=" + str(j)
                    if training_df is None:
                        training_df = folds[j]
                    else:
                        training_df = training_df.append(folds[j])

            print "Run training gibbs " + str(training_df.shape)
            training_gibbs = CollapseGibbsLda(training_df, self.K, self.alpha, self.beta, 
                                              silent=False)
            training_gibbs.run(n_burn, n_samples, n_thin, use_native=True)
            
            print "Run testing gibbs " + str(testing_df.shape)
            testing_gibbs = CollapseGibbsLda(testing_df, self.K, self.alpha, self.beta, 
                                             previous_model=training_gibbs, silent=False)
            testing_gibbs.run(n_burn, n_samples, n_thin, use_native=True)
        
            # testing_hm = stats.hmean(testing_gibbs.all_lls)
            testing_hm = len(testing_gibbs.all_lls) / np.sum(1.0/testing_gibbs.all_lls) 
            print "Harmonic mean for testing fold " + str(testing_idx) + " = " + str(testing_hm)
            print
            testing_hms.append(testing_hm)
            
        testing_hm = np.array(testing_hms)
        mean_marg = np.mean(testing_hm)
        self.mean_marg = np.asscalar(mean_marg)
        print
        print "Cross-validation done!"
        print "K=" + str(self.K) + ", mean_approximate_log_marginal_likelihood=" + str(self.mean_marg)
    

def run_cv(df, k, alpha, beta):    

    cv = CrossValidatorLda(df, k, alpha, beta)
    cv.cross_validate(n_folds=4, n_burn=100, n_samples=200, n_thin=5)    
    return cv.mean_marg

def run_synthetic(parallel=True):

    K = 10
    print "Cross-validation for K=" + str(K)
    alpha = 0.1
    beta = 0.01    
    n_docs = 50
    vocab_size = 500
    document_length = 50
    gen = LdaDataGenerator(alpha)
    df = gen.generate_input_df(K, vocab_size, document_length, n_docs)
    
    ks = range(1, 20, 1)
    if parallel:
        num_cores = multiprocessing.cpu_count()
        mean_margs = Parallel(n_jobs=num_cores)(delayed(run_cv)(df, k, alpha, beta) for k in ks)      
    else:
        mean_margs = []
        for k in ks:
            mean_marg = run_cv(df, k, alpha, beta)
            mean_margs.append(mean_marg)
        
    plt.figure()
    plt.plot(np.array(ks), np.array(mean_margs))
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('Marg')
    plt.title('CV results')
    plt.show()

def run_beer3():

    if len(sys.argv)>1:
        K = int(sys.argv[1])
    else:
        K = 250

    # find the current path of this script file        
    current_path = os.path.dirname(os.path.abspath(__file__))
        
    print "Cross-validation for K=" + str(K)
    n_folds = 4
    n_samples = 500
    n_burn = 250
    n_thin = 5
    alpha = 0.1
    beta = 0.01
     
    relative_intensity = True
    fragment_filename = current_path + '/input/relative_intensities/Beer_3_T10_POS_fragments_rel.csv'
    neutral_loss_filename = current_path + '/input/relative_intensities/Beer_3_T10_POS_losses_rel.csv'
    mzdiff_filename = None
    ms1_filename = current_path + '/input/relative_intensities/Beer_3_T10_POS_ms1_rel.csv'
    ms2_filename = current_path + '/input/relative_intensities/Beer_3_T10_POS_ms2_rel.csv'
    ms2lda = Ms2Lda(fragment_filename, neutral_loss_filename, mzdiff_filename,
                ms1_filename, ms2_filename, relative_intensity)
     
    df = ms2lda.preprocess()
    cv = CrossValidatorLda(df, K, alpha, beta)
    cv.cross_validate(n_folds, n_burn, n_samples, n_thin)    

def run_urine37():

    if len(sys.argv)>1:
        K = int(sys.argv[1])
    else:
        K = 250
        
    # find the current path of this script file        
    current_path = os.path.dirname(os.path.abspath(__file__))        
        
    print "Cross-validation for K=" + str(K)
    n_folds = 4
    n_samples = 500
    n_burn = 250
    n_thin = 5
    alpha = 0.1
    beta = 0.01
     
    relative_intensity = True
    fragment_filename = current_path + '/input/relative_intensities/Urine_37_Top10_POS_fragments_rel.csv'
    neutral_loss_filename = current_path + '/input/relative_intensities/Urine_37_Top10_POS_losses_rel.csv'
    mzdiff_filename = None
    ms1_filename = current_path + '/input/relative_intensities/Urine_37_Top10_POS_ms1_rel.csv'
    ms2_filename = current_path + '/input/relative_intensities/Urine_37_Top10_POS_ms2_rel.csv'
    ms2lda = Ms2Lda(fragment_filename, neutral_loss_filename, mzdiff_filename,
                ms1_filename, ms2_filename, relative_intensity)
     
    df = ms2lda.preprocess()
    cv = CrossValidatorLda(df, K, alpha, beta)
    cv.cross_validate(n_folds, n_burn, n_samples, n_thin)    

def main():    

    data = None
    if len(sys.argv)>2:
        data = sys.argv[2].upper()

    if data == 'BEER3POS':
        print "Data = Beer3 Positive"
        run_beer3()
    elif data == 'URINE37POS':
        print "Data = Urine37 Positive"
        run_urine37()
    else:
        print "Data = Synthetic"
        run_synthetic(parallel=False)        

if __name__ == "__main__":
    main()
