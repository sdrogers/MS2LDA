"""
Cross-validation for LDA
"""

import multiprocessing
import os
import sys

from joblib import Parallel, delayed  

from lda_is import ldae_is_variants
from lda_cgs import CollapseGibbsLda
from lda_for_fragments import Ms2Lda
from lda_generate_data import LdaDataGenerator
import lda_utils as utils
import numpy as np
import pylab as plt
from collections import namedtuple

Cv_Results = namedtuple('Cv_Results', 'marg perp')
class CrossValidatorLda:
    
    def __init__(self, df, vocab, K, alpha, beta):
        self.df = df
        self.vocab = vocab
        self.K = K
        self.alpha = alpha
        self.beta = beta

    # run cross-validation by using the importance sampling approximation of the marginal likelihood on the testing set 
    def cross_validate_is(self, n_folds, n_burn, n_samples, n_thin,
                          is_num_samples, is_iters):
    
        shuffled_df = self.df.reindex(np.random.permutation(self.df.index))
        folds = np.array_split(shuffled_df, n_folds)
                
        margs = []
        test_perplexity = []
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
            training_gibbs = CollapseGibbsLda(training_df, self.vocab, self.K, self.alpha, self.beta)
            training_gibbs.run(n_burn, n_samples, n_thin, use_native=True)
            
            print "Run testing importance sampling " + str(testing_df.shape)
            topics = training_gibbs.topic_word_
            # use posterior alpha instead of prior, this is inferred from the last sample of training_gibbs
#             topic_prior = np.ones((self.K, 1))
#             topic_prior = topic_prior / np.sum(topic_prior)            
#             topic_prior = topic_prior * self.K * self.alpha
            topic_prior = training_gibbs.posterior_alpha[:, None]
            print 'topic_prior = ' + str(topic_prior)
            marg = 0         
            n_words = 0
            for d in range(testing_df.shape[0]):
                document = self.df.iloc[[d]]
                words = utils.word_indices(document)
                doc_marg = ldae_is_variants(words, topics, topic_prior, 
                                         num_samples=is_num_samples, variant=3, variant_iters=is_iters)
                print "\td = " + str(d) + " doc_marg=" + str(doc_marg)
                sys.stdout.flush()                
                marg += doc_marg              
                n_words += len(words)

            perp = np.exp(-(marg/n_words))
            print "Log evidence " + str(testing_idx) + " = " + str(marg)
            print "Test perplexity " + str(testing_idx) + " = " + str(perp)
            print
            margs.append(marg)
            test_perplexity.append(perp)
            
        margs = np.array(marg)
        mean_marg = np.mean(margs)
        self.mean_margs = np.asscalar(mean_marg)

        test_perplexity = np.array(test_perplexity)
        mean_perplexity = np.mean(test_perplexity)
        self.mean_perplexities = np.asscalar(mean_perplexity)        
        
        print
        print "Cross-validation done!"
        print "K=" + str(self.K) + ",mean_approximate_log_evidence=" + str(self.mean_margs) \
            + ",mean_perplexity=" + str(self.mean_perplexities)  

def run_cv(df, vocab, k, alpha, beta):    

    cv = CrossValidatorLda(df, vocab, k, alpha, beta)
    cv.cross_validate_is(n_folds=4, n_burn=0, n_samples=200, n_thin=1, 
                         is_num_samples=1000, is_iters=1000)
    
    res = Cv_Results(cv.mean_margs, cv.mean_perplexities)
    return res

def run_synthetic(parallel=True):

    K = 50
    print "Cross-validation for K=" + str(K)
#     alpha = 50.0/K
#     beta = 0.1
    alpha = 0.1
    beta = 0.01    
    n_docs = 200
    vocab_size = 500
    document_length = 50
    gen = LdaDataGenerator(alpha)
    df, vocab = gen.generate_input_df(K, vocab_size, document_length, n_docs)
    
    ks = range(10, 101, 10)
    if parallel:
        num_cores = multiprocessing.cpu_count()
        res = Parallel(n_jobs=num_cores)(delayed(run_cv)(df, vocab, k, alpha, beta) for k in ks)      
        mean_margs = []
        mean_perplexities = []
        for r in res:
            mean_margs.append(r.marg)
            mean_perplexities.append(r.perp)
    else:
        mean_margs = []
        mean_perplexities = []
        for k in ks:
            mean_marg, mean_perplexity = run_cv(df, vocab, k, alpha, beta)
            mean_margs.append(mean_marg)
            mean_perplexities.append(mean_perplexity)
        
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(np.array(ks), np.array(mean_margs))
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('Log evidence')
    plt.title('CV results -- log evidence for varying K')

    plt.subplot(1, 2, 2)
    plt.plot(np.array(ks), np.array(mean_perplexities))
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('Perplexity')
    plt.title('CV results -- perplexity for varying K')

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
    n_samples = 10
    n_burn = 0
    n_thin = 1
    alpha = 50.0/K
    beta = 0.1
    is_num_samples = 1000
    is_iters = 1
     
    relative_intensity = True
    fragment_filename = current_path + '/input/relative_intensities/Beer_3_T10_POS_fragments_rel.csv'
    neutral_loss_filename = current_path + '/input/relative_intensities/Beer_3_T10_POS_losses_rel.csv'
    mzdiff_filename = None
    ms1_filename = current_path + '/input/relative_intensities/Beer_3_T10_POS_ms1_rel.csv'
    ms2_filename = current_path + '/input/relative_intensities/Beer_3_T10_POS_ms2_rel.csv'
    ms2lda = Ms2Lda(fragment_filename, neutral_loss_filename, mzdiff_filename,
                ms1_filename, ms2_filename, relative_intensity)
     
    df, vocab = ms2lda.preprocess()
    cv = CrossValidatorLda(df, vocab, K, alpha, beta)
    cv.cross_validate_is(n_folds, n_burn, n_samples, n_thin, 
                         is_num_samples, is_iters)         

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
    n_burn = 0
    n_thin = 1
    alpha = 50.0/K
    beta = 0.1
    is_num_samples = 1000
    is_iters = 1000
     
    relative_intensity = True
    fragment_filename = current_path + '/input/relative_intensities/Urine_37_Top10_POS_fragments_rel.csv'
    neutral_loss_filename = current_path + '/input/relative_intensities/Urine_37_Top10_POS_losses_rel.csv'
    mzdiff_filename = None
    ms1_filename = current_path + '/input/relative_intensities/Urine_37_Top10_POS_ms1_rel.csv'
    ms2_filename = current_path + '/input/relative_intensities/Urine_37_Top10_POS_ms2_rel.csv'
    ms2lda = Ms2Lda(fragment_filename, neutral_loss_filename, mzdiff_filename,
                ms1_filename, ms2_filename, relative_intensity)
     
    df, vocab = ms2lda.preprocess()
    cv = CrossValidatorLda(df, vocab, K, alpha, beta)
    cv.cross_validate_is(n_folds, n_burn, n_samples, n_thin, 
                         is_num_samples, is_iters)         

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
