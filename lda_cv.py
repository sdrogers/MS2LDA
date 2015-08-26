"""
Cross-validation for LDA and multinomial mixture model
"""
from collections import namedtuple
import multiprocessing
import os
import sys

from joblib import Parallel, delayed  

from lda_cgs import CollapseGibbsLda
from lda_for_fragments import Ms2Lda
from lda_generate_data import LdaDataGenerator
from lda_is import ldae_is_variants
from mixture import mixture_cgs
import lda_utils as utils
import matplotlib.patches as mpatches
import numpy as np
import pylab as plt


Cv_Results = namedtuple('Cv_Results', 'training_lda_marg, training_lda_perp \
                                        training_mixture_marg, training_mixture_perp, \
                                        testing_lda_fold_in_marg testing_lda_fold_in_perp \
                                        testing_lda_is_marg testing_lda_is_perp \
                                        testing_mixture_fold_in_marg testing_mixture_fold_in_perp')
class CrossValidatorLda:
    
    def __init__(self, df, vocab, K, alpha, beta):
        self.df = df
        self.vocab = vocab
        self.K = K
        self.alpha = alpha
        self.beta = beta

    def cross_validate(self, n_folds, n_burn, n_samples, n_thin,
                          is_num_samples, is_iters, method="both"):

        folds = self._make_folds(n_folds)

        # training results
        training_lda_margs = []
        training_lda_perps = []
        training_mixture_margs = []
        training_mixture_perps = []

        # testing results
        testing_lda_fold_in_margs = []
        testing_lda_fold_in_perps = []
        testing_lda_is_margs = []
        testing_lda_is_perps = []
        testing_mixture_fold_in_margs = []
        testing_mixture_fold_in_perps = []
        
        for i in range(len(folds)):

            # vary the training-testing folds each time            
            training_df = None
            testing_df = None
            training_idx = None
            testing_idx = None
            for j in range(len(folds)):
                if j == i:
                    print "K=" + str(self.K) + " Testing fold=" + str(j)
                    testing_df = folds[j]
                    testing_idx = j
                else:
                    print "K=" + str(self.K) + " Training fold=" + str(j)
                    if training_df is None:
                        training_df = folds[j]
                        training_idx = [j]
                    else:
                        training_df = training_df.append(folds[j])
                        training_idx.append(j)

            # run training LDA on the training fold
            training_gibbs, training_marg, training_perp = self._train_lda(training_df, training_idx, 
                                                                           n_burn, n_samples, n_thin)
            training_lda_margs.append(training_marg)
            training_lda_perps.append(training_perp)            
 
            # get testing performance using the fold-in method (holding the topic-word distribution fixed)
            testing_marg, testing_perp = self._test_lda_fold_in(testing_df, testing_idx, 
                                                                n_burn, n_samples, n_thin, 
                                                                training_gibbs)
            testing_lda_fold_in_margs.append(testing_marg)
            testing_lda_fold_in_perps.append(testing_perp)
             
            # get testing performance using pseudo-count importance sampling in
            # Wallach, Hanna M., et al. "Evaluation methods for topic models." 
            # Proceedings of the 26th Annual International Conference on Machine Learning. ACM, 2009.
            testing_marg, testing_perp = self._test_lda_importance_sampling(testing_df, testing_idx, 
                                                                            is_num_samples, is_iters, training_gibbs, 
                                                                            use_posterior_alpha=True)
            testing_lda_is_margs.append(testing_marg)
            testing_lda_is_perps.append(testing_perp)

            if method == "with_mixture":

                # run training multinomial mixture on the training fold
                training_gibbs, training_marg, training_perp = self._train_mixture(training_df, training_idx, 
                                                                                   n_burn, n_samples, n_thin)
                training_mixture_margs.append(training_marg)
                training_mixture_perps.append(training_perp)
    
                # get testing performance using the fold-in method
                testing_marg, testing_perp = self._test_mixture_fold_in(testing_df, testing_idx, 
                                                                        n_burn, n_samples, n_thin, 
                                                                        training_gibbs)
                testing_mixture_fold_in_margs.append(testing_marg)
                testing_mixture_fold_in_perps.append(testing_perp)

        # average training results across all folds
        training_lda_marg, training_lda_perp  = self._get_all_folds_performance(training_lda_margs, training_lda_perps)
        training_mixture_marg, training_mixture_perp  = self._get_all_folds_performance(training_mixture_margs, training_mixture_perps)

        # average testing results across all folds
        testing_lda_fold_in_marg, testing_lda_fold_in_perp  = self._get_all_folds_performance(testing_lda_fold_in_margs, 
                                                                                              testing_lda_fold_in_perps)
        testing_lda_is_marg, testing_lda_is_perp  = self._get_all_folds_performance(testing_lda_is_margs, 
                                                                                    testing_lda_is_perps)
        testing_mixture_fold_in_marg, testing_mixture_fold_in_perp  = self._get_all_folds_performance(testing_mixture_fold_in_margs, 
                                                                                                      testing_mixture_fold_in_perps)
        
        print
        print "K=" + str(self.K) \
            + ",training_lda_log_evidence=" + str(training_lda_marg) \
            + ",training_lda_perplexity=" + str(training_lda_perp) \
            + ",training_mixture_log_evidence=" + str(training_mixture_marg) \
            + ",training_mixture_perplexity=" + str(training_mixture_perp) \
            + ",testing_lda_fold_in_log_evidence=" + str(testing_lda_fold_in_marg) \
            + ",testing_lda_fold_in_perplexity=" + str(testing_lda_fold_in_perp) \
            + ",testing_lda_importance_sampling_evidence=" + str(testing_lda_is_marg) \
            + ",testing_lda_importance_sampling_perplexity=" + str(testing_lda_is_perp) \
            + ",testing_mixture_fold_in_log_evidence=" + str(testing_mixture_fold_in_marg) \
            + ",testing_mixture_fold_in_perplexity=" + str(testing_mixture_fold_in_perp)

        res = Cv_Results(training_lda_marg, training_lda_perp,
                         training_mixture_marg, training_mixture_perp,
                         testing_lda_fold_in_marg, testing_lda_fold_in_perp, 
                         testing_lda_is_marg, testing_lda_is_perp, 
                         testing_mixture_fold_in_marg, testing_mixture_fold_in_perp)
        return res

    def _make_folds(self, n_folds):
        
        shuffled_df = self.df.reindex(np.random.permutation(self.df.index))
        folds = np.array_split(shuffled_df, n_folds)
        return folds
    
    def _train_lda(self, training_df, fold_idx, n_burn, n_samples, n_thin):

        print "Run training gibbs " + str(training_df.shape)
        training_gibbs = CollapseGibbsLda(training_df, self.vocab, self.K, self.alpha, self.beta)
        training_gibbs.run(n_burn, n_samples, n_thin, use_native=True)
        marg, perp = self._average_samples("lda fold-in training", fold_idx, training_gibbs)
        return training_gibbs, marg, perp

    def _test_lda_fold_in(self, testing_df, fold_idx, n_burn, n_samples, n_thin, training_gibbs):

        print "Run testing gibbs " + str(testing_df.shape)
        testing_gibbs = CollapseGibbsLda(testing_df, self.vocab, self.K, self.alpha, self.beta, 
                                         previous_model=training_gibbs)
        testing_gibbs.run(n_burn, n_samples, n_thin, use_native=True)  
        marg, perp = self._average_samples("lda fold-in testing", fold_idx, testing_gibbs)
        return marg, perp

    def _test_lda_importance_sampling(self, testing_df, fold_idx, 
                                      is_num_samples, is_iters, 
                                      training_gibbs, use_posterior_alpha=True):

        print "Run testing importance sampling " + str(testing_df.shape)
        topics = training_gibbs.topic_word_

        if use_posterior_alpha:
            # use posterior alpha as the topic prior during importance sampling
            topic_prior = training_gibbs.posterior_alpha[:, None]
        else:
            # use prior alpha as the topic prior during importance sampling
            topic_prior = np.ones((self.K, 1))
            topic_prior = topic_prior / np.sum(topic_prior)            
            topic_prior = topic_prior * self.K * self.alpha

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

        print "lda is testing log evidence fold " + str(fold_idx) + " = " + str(marg)
        print "lda is testing perplexity fold " + str(fold_idx) + " = " + str(perp)
        print
        return marg, perp

    def _train_mixture(self, training_df, fold_idx, n_burn, n_samples, n_thin):

        print "Run training gibbs " + str(training_df.shape)
        training_gibbs = mixture_cgs.CollapseGibbsMixture(training_df, self.vocab, self.K, self.alpha, self.beta)
        training_gibbs.run(n_burn, n_samples, n_thin, use_native=True)
        marg, perp = self._average_samples("mixture fold-in training", fold_idx, training_gibbs)
        return training_gibbs, marg, perp

    def _test_mixture_fold_in(self, testing_df, fold_idx, n_burn, n_samples, n_thin, training_gibbs):

        print "Run testing gibbs " + str(testing_df.shape)
        testing_gibbs = mixture_cgs.CollapseGibbsMixture(testing_df, self.vocab, self.K, self.alpha, self.beta, 
                                         previous_model=training_gibbs)
        testing_gibbs.run(n_burn, n_samples, n_thin, use_native=True)  
        marg, perp = self._average_samples("mixture fold-in testing", fold_idx, testing_gibbs)
        return marg, perp
    
    def _average_samples(self, prefix, fold_idx, gibbs):

        # average over the evidences and perplexities in the samples
        sample_margs = np.array(gibbs.margs)
        sample_perps = np.array(gibbs.perps)
        avg_marg = np.mean(sample_margs)
        avg_perp = np.mean(sample_perps)

        print prefix + " log evidence fold " + str(fold_idx) + " = " + str(avg_marg)
        print prefix + " perplexity fold " + str(fold_idx) + " = " + str(avg_perp)
        print
        return avg_marg, avg_perp
    
    def _get_all_folds_performance(self, margs, perps):
        margs = np.array(margs)
        perps = np.array(perps)
        avg_marg = np.asscalar(np.mean(margs))
        avg_perp = np.asscalar(np.mean(perps))
        return avg_marg, avg_perp

def main():    

    data = None
    if len(sys.argv)>2:
        data = sys.argv[2].upper()

    # find the current path of this script file        
    current_path = os.path.dirname(os.path.abspath(__file__))

    if data == 'BEER3POS':

        print "Data = Beer3 Positive"        
        fragment = current_path + '/input/final/Beer_3_full1_5_2E5_pos_fragments.csv'
        loss = current_path + '/input/final/Beer_3_full1_5_2E5_pos_losses.csv'
        mzdiff = None    
        ms1 = current_path + '/input/final/Beer_3_full1_5_2E5_pos_ms1.csv'
        ms2 = current_path + '/input/final/Beer_3_full1_5_2E5_pos_ms2.csv'  
        run_msms_data(fragment, loss, mzdiff, ms1, ms2)

    elif data == 'URINE37POS':

        print "Data = Urine37 Positive"        
        fragment = current_path + '/input/final/Urine_64_fullscan1_5_2E5_POS_fragments.csv'
        loss = current_path + '/input/final/Urine_64_fullscan1_5_2E5_POS_losses.csv'
        mzdiff = None    
        ms1 = current_path + '/input/final/Urine_64_fullscan1_5_2E5_POS_ms1.csv'
        ms2 = current_path + '/input/final/Urine_64_fullscan1_5_2E5_POS_ms2.csv'  
        run_msms_data(fragment, loss, mzdiff, ms1, ms2)

    elif data == 'STD1POS':
    
        print "Data = Standard Mix 1 Positive"
        fragment = current_path + '/input/relative_intensities/STD_MIX1_POS_60stepped_1E5_Top5_fragments_rel.csv'
        loss = current_path + '/input/relative_intensities/STD_MIX1_POS_60stepped_1E5_Top5_losses_rel.csv'
        mzdiff = None
        ms1 = current_path + '/input/relative_intensities/STD_MIX1_POS_60stepped_1E5_Top5_ms1_rel.csv'
        ms2 = current_path + '/input/relative_intensities/STD_MIX1_POS_60stepped_1E5_Top5_ms2_rel.csv'
        run_msms_data(fragment, loss, mzdiff, ms1, ms2)

    else:
    
        print "Data = Synthetic"
        run_synthetic(parallel=False)        

def run_msms_data(fragment, neutral_loss, mzdiff, 
                  ms1, ms2):

    if len(sys.argv)>1:
        K = int(sys.argv[1])
    else:
        K = 300
        
    print "Cross-validation for K=" + str(K)
    n_folds = 4
    n_samples = 500
    n_burn = 250
    n_thin = 5
    alpha = 50.0/K
    beta = 0.1
    is_num_samples = 10000
    is_iters = 1000
     
    ms2lda = Ms2Lda.lcms_data_from_R(fragment, neutral_loss, mzdiff, ms1, ms2)    
    df = ms2lda.df
    vocab = ms2lda.vocab
    cv = CrossValidatorLda(df, vocab, K, alpha, beta)
    cv.cross_validate(n_folds, n_burn, n_samples, n_thin, 
                         is_num_samples, is_iters, method="with_mixture")         

def run_synthetic(parallel=True):

    K = 50
    print "Cross-validation for K=" + str(K)
    alpha = 0.1
    beta = 0.01    
    n_docs = 200
    vocab_size = 500
    document_length = 50
    gen = LdaDataGenerator(alpha)
    df, vocab = gen.generate_input_df(K, vocab_size, document_length, n_docs)
    
    ks = range(10, 101, 10)
    training_lda_margs = []
    training_lda_perps = []
    training_mixture_margs = []
    training_mixture_perps = []
    testing_lda_fold_in_margs = []
    testing_lda_fold_in_perps = []
    testing_lda_is_margs = []
    testing_lda_is_perps = []
    testing_mixture_fold_in_margs = []
    testing_mixture_fold_in_perps = []
    if parallel:
        num_cores = multiprocessing.cpu_count()
        res = Parallel(n_jobs=num_cores)(delayed(run_cv)(df, vocab, k, alpha, beta) for k in ks)      
        for r in res:
            training_lda_margs.append(r.training_lda_marg)
            training_lda_perps.append(r.training_lda_perp)
            training_mixture_margs.append(r.training_mixture_marg)
            training_mixture_perps.append(r.training_mixture_perp)
            testing_lda_fold_in_margs.append(r.testing_lda_fold_in_marg)
            testing_lda_fold_in_perps.append(r.testing_lda_fold_in_perp)
            testing_lda_is_margs.append(r.testing_lda_is_marg)
            testing_lda_is_perps.append(r.testing_lda_is_perp)
            testing_mixture_fold_in_margs.append(r.testing_mixture_fold_in_marg)
            testing_mixture_fold_in_perps.append(r.testing_mixture_fold_in_perp)
    else:
        for k in ks:
            r = run_cv(df, vocab, k, alpha, beta)
            training_lda_margs.append(r.training_lda_marg)
            training_lda_perps.append(r.training_lda_perp)
            training_mixture_margs.append(r.training_mixture_marg)
            training_mixture_perps.append(r.training_mixture_perp)
            testing_lda_fold_in_margs.append(r.testing_lda_fold_in_marg)
            testing_lda_fold_in_perps.append(r.testing_lda_fold_in_perp)
            testing_lda_is_margs.append(r.testing_lda_is_marg)
            testing_lda_is_perps.append(r.testing_lda_is_perp)
            testing_mixture_fold_in_margs.append(r.testing_mixture_fold_in_marg)
            testing_mixture_fold_in_perps.append(r.testing_mixture_fold_in_perp)
            
    _make_training_plot(ks, training_lda_margs, training_mixture_margs, training_lda_perps, training_mixture_perps)
    _make_testing_plot(ks, testing_lda_fold_in_margs, testing_lda_is_margs, testing_mixture_fold_in_margs,
                       testing_lda_fold_in_perps, testing_lda_is_perps, testing_mixture_fold_in_perps)
        
def _make_training_plot(ks, lda_margs, mixture_margs, lda_perps, mixture_perps):

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(np.array(ks), np.array(lda_margs), 'r')
    plt.plot(np.array(ks), np.array(mixture_margs), 'b')
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('Log evidence')
    plt.title('Log evidence')

    plt.subplot(1, 2, 2)
    plt.plot(np.array(ks), np.array(lda_perps), 'r')
    plt.plot(np.array(ks), np.array(mixture_perps), 'b')
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('Perplexity')
    plt.title('Perplexity')

    red_patch = mpatches.Patch(color='red', label='LDA')
    blue_patch = mpatches.Patch(color='blue', label='Mixture')
    plt.suptitle("Training Performance")
    plt.legend(handles=[red_patch, blue_patch])                
    plt.tight_layout()
    plt.show()    
    
def _make_testing_plot(ks, 
               fold_in_margs, is_margs, mixture_margs,
               fold_in_perps, is_perps, mixture_perps):

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(np.array(ks), np.array(fold_in_margs), 'r')
    plt.plot(np.array(ks), np.array(is_margs), 'g')
    plt.plot(np.array(ks), np.array(mixture_margs), 'b')
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('Log evidence')
    plt.title('Log evidence')

    plt.subplot(1, 2, 2)
    plt.plot(np.array(ks), np.array(fold_in_perps), 'r')
    plt.plot(np.array(ks), np.array(is_perps), 'g')
    plt.plot(np.array(ks), np.array(mixture_perps), 'b')
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('Perplexity')
    plt.title('Perplexity')

    red_patch = mpatches.Patch(color='red', label='LDA Fold-in')
    green_patch = mpatches.Patch(color='green', label='LDA IS')
    blue_patch = mpatches.Patch(color='blue', label='Mixture Fold-in')
    plt.suptitle("Testing Performance")
    plt.legend(handles=[red_patch, green_patch, blue_patch])                
    plt.tight_layout()
    plt.show()    

def run_cv(df, vocab, k, alpha, beta):    

    cv = CrossValidatorLda(df, vocab, k, alpha, beta)
    res = cv.cross_validate(n_folds=4, n_burn=250, n_samples=500, n_thin=5, 
                         is_num_samples=10000, is_iters=1000, method="with_mixture")
    return res

if __name__ == "__main__":
    main()
