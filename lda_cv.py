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
from justin.mixture import mixture_cgs
import lda_utils as utils
import matplotlib.patches as mpatches
import numpy as np
import pylab as plt


Cv_Results = namedtuple('Cv_Results', 'lda_fold_in_marg lda_fold_in_perp \
                                        lda_is_marg lda_is_perp \
                                        mixture_fold_in_marg mixture_fold_in_perp')
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

        lda_fold_in_margs = []
        lda_fold_in_perps = []
        lda_is_margs = []
        lda_is_perps = []
        mixture_fold_in_margs = []
        mixture_fold_in_perps = []
        
        for i in range(len(folds)):

            # vary the training-testing folds each time            
            training_df = None
            testing_df = None
            testing_idx = i
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

            # run training LDA on the training fold
            training_gibbs = self._train_lda(training_df, n_burn, n_samples, n_thin)

            # get testing performance using the fold-in method (holding the topic-word distribution fixed)
            testing_marg, testing_perp = self._test_lda_fold_in(testing_df, testing_idx, 
                                                                n_burn, n_samples, n_thin, 
                                                                training_gibbs)
            lda_fold_in_margs.append(testing_marg)
            lda_fold_in_perps.append(testing_perp)
            
            # get testing performance using pseudo-count importance sampling in
            # Wallach, Hanna M., et al. "Evaluation methods for topic models." 
            # Proceedings of the 26th Annual International Conference on Machine Learning. ACM, 2009.
            testing_marg, testing_perp = self._test_lda_importance_sampling(testing_df, testing_idx, 
                                                                            is_num_samples, is_iters, training_gibbs, 
                                                                            use_posterior_alpha=True)
            lda_is_margs.append(testing_marg)
            lda_is_perps.append(testing_perp)

            if method == "with_mixture":

                # run training multinomial mixture on the training fold
                training_gibbs = self._train_mixture(training_df, n_burn, n_samples, n_thin)
    
                # get testing performance using the fold-in method
                testing_marg, testing_perp = self._test_mixture_fold_in(testing_df, testing_idx, 
                                                                        n_burn, n_samples, n_thin, 
                                                                        training_gibbs)
                mixture_fold_in_margs.append(testing_marg)
                mixture_fold_in_perps.append(testing_perp)

        # outside the for loop of the folds

        lda_fold_in_margs = np.array(lda_fold_in_margs)
        lda_fold_in_perps = np.array(lda_fold_in_perps)
        avg_lda_fold_in_marg = np.asscalar(np.mean(lda_fold_in_margs))
        avg_lda_fold_in_perp = np.asscalar(np.mean(lda_fold_in_perps))

        lda_is_margs = np.array(lda_is_margs)
        lda_is_perps = np.array(lda_is_perps)
        avg_lda_is_marg = np.asscalar(np.mean(lda_is_margs))
        avg_lda_is_perp = np.asscalar(np.mean(lda_is_perps))

        mixture_fold_in_margs = np.array(mixture_fold_in_margs)
        mixture_fold_in_perps = np.array(mixture_fold_in_perps)
        avg_mixture_fold_in_marg = np.asscalar(np.mean(mixture_fold_in_margs))
        avg_mixture_fold_in_perp = np.asscalar(np.mean(mixture_fold_in_perps))
        
        print
        print "K=" + str(self.K) \
            + ",lda_fold_in_log_evidence=" + str(avg_lda_fold_in_marg) \
            + ",lda_fold_in_perplexity=" + str(avg_lda_fold_in_perp) \
            + ",lda_importance_sampling_evidence=" + str(avg_lda_is_marg) \
            + ",lda_importance_sampling_perplexity=" + str(avg_lda_is_perp) \
            + ",mixture_fold_in_log_evidence=" + str(avg_mixture_fold_in_marg) \
            + ",mixture_fold_in_perplexity=" + str(avg_mixture_fold_in_perp)

        res = Cv_Results(avg_lda_fold_in_marg, avg_lda_fold_in_perp, 
                         avg_lda_is_marg, avg_lda_is_perp, 
                         avg_mixture_fold_in_marg, avg_mixture_fold_in_perp)
        return res

    def _make_folds(self, n_folds):
        
        shuffled_df = self.df.reindex(np.random.permutation(self.df.index))
        folds = np.array_split(shuffled_df, n_folds)
        return folds
    
    def _train_lda(self, training_df, n_burn, n_samples, n_thin):

        print "Run training gibbs " + str(training_df.shape)
        training_gibbs = CollapseGibbsLda(training_df, self.vocab, self.K, self.alpha, self.beta)
        training_gibbs.run(n_burn, n_samples, n_thin, use_native=True)
        return training_gibbs

    def _test_lda_fold_in(self, testing_df, testing_idx, n_burn, n_samples, n_thin, training_gibbs):

        print "Run testing gibbs " + str(testing_df.shape)
        testing_gibbs = CollapseGibbsLda(testing_df, self.vocab, self.K, self.alpha, self.beta, 
                                         previous_model=training_gibbs)
        testing_gibbs.run(n_burn, n_samples, n_thin, use_native=True)  
        
        # average over the evidences and perplexities in the samples
        sample_margs = np.array(testing_gibbs.margs)
        sample_perps = np.array(testing_gibbs.perps)
        avg_marg = np.mean(sample_margs)
        avg_perp = np.mean(sample_perps)

        print "Log evidence " + str(testing_idx) + " = " + str(avg_marg)
        print "Test perplexity " + str(testing_idx) + " = " + str(avg_perp)
        print
        return avg_marg, avg_perp

    def _test_lda_importance_sampling(self, testing_df, testing_idx, 
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
        print "Log evidence " + str(testing_idx) + " = " + str(marg)
        print "Test perplexity " + str(testing_idx) + " = " + str(perp)
        print
        return marg, perp

    def _train_mixture(self, training_df, n_burn, n_samples, n_thin):

        print "Run training gibbs " + str(training_df.shape)
        training_gibbs = mixture_cgs.CollapseGibbsMixture(training_df, self.vocab, self.K, self.alpha, self.beta)
        training_gibbs.run(n_burn, n_samples, n_thin, use_native=True)
        return training_gibbs

    def _test_mixture_fold_in(self, testing_df, testing_idx, n_burn, n_samples, n_thin, training_gibbs):

        print "Run testing gibbs " + str(testing_df.shape)
        testing_gibbs = mixture_cgs.CollapseGibbsMixture(testing_df, self.vocab, self.K, self.alpha, self.beta, 
                                         previous_model=training_gibbs)
        testing_gibbs.run(n_burn, n_samples, n_thin, use_native=True)  
        
        # average over the evidences and perplexities in the samples
        sample_margs = np.array(testing_gibbs.margs)
        sample_perps = np.array(testing_gibbs.perps)
        avg_marg = np.mean(sample_margs)
        avg_perp = np.mean(sample_perps)

        print "Log evidence " + str(testing_idx) + " = " + str(avg_marg)
        print "Test perplexity " + str(testing_idx) + " = " + str(avg_perp)
        print
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
        run_synthetic(parallel=True)        

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
                         is_num_samples, is_iters)         

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
    lda_fold_in_margs = []
    lda_fold_in_perps = []
    lda_is_margs = []
    lda_is_perps = []
    mixture_fold_in_margs = []
    mixture_fold_in_perps = []
    if parallel:
        num_cores = multiprocessing.cpu_count()
        res = Parallel(n_jobs=num_cores)(delayed(run_cv)(df, vocab, k, alpha, beta) for k in ks)      
        for r in res:
            lda_fold_in_margs.append(r.lda_fold_in_marg)
            lda_fold_in_perps.append(r.lda_fold_in_perp)
            lda_is_margs.append(r.lda_is_marg)
            lda_is_perps.append(r.lda_is_perp)
            mixture_fold_in_margs.append(r.mixture_fold_in_marg)
            mixture_fold_in_perps.append(r.mixture_fold_in_perp)
    else:
        for k in ks:
            r = run_cv(df, vocab, k, alpha, beta)
            lda_fold_in_margs.append(r.lda_fold_in_marg)
            lda_fold_in_perps.append(r.lda_fold_in_perp)
            lda_is_margs.append(r.lda_is_marg)
            lda_is_perps.append(r.lda_is_perp)
            mixture_fold_in_margs.append(r.mixture_fold_in_marg)
            mixture_fold_in_perps.append(r.mixture_fold_in_perp)
        
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(np.array(ks), np.array(lda_fold_in_margs), 'r')
    plt.plot(np.array(ks), np.array(lda_is_margs), 'g')
    plt.plot(np.array(ks), np.array(mixture_fold_in_margs), 'b')
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('Log evidence')
    plt.title('CV results -- log evidence for varying K')

    plt.subplot(1, 2, 2)
    plt.plot(np.array(ks), np.array(lda_fold_in_perps), 'r')
    plt.plot(np.array(ks), np.array(lda_is_perps), 'g')
    plt.plot(np.array(ks), np.array(mixture_fold_in_perps), 'b')
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('Perplexity')
    plt.title('CV results -- perplexity for varying K')

    red_patch = mpatches.Patch(color='red', label='LDA Fold-in')
    green_patch = mpatches.Patch(color='green', label='LDA IS')
    blue_patch = mpatches.Patch(color='blue', label='LDA IS')
    plt.legend(handles=[red_patch, green_patch, blue_patch])                
    plt.show()

def run_cv(df, vocab, k, alpha, beta):    

    cv = CrossValidatorLda(df, vocab, k, alpha, beta)
    res = cv.cross_validate(n_folds=4, n_burn=250, n_samples=500, n_thin=5, 
                         is_num_samples=10000, is_iters=1000, method="with_mixture")
    return res

if __name__ == "__main__":
    main()
