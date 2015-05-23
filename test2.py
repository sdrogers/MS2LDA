"""
Implementation of collapsed Gibbs sampling for LDA
"""

import sys
import time

from lda_cgs import CollapseGibbsLda
from lda_for_fragments import Ms2Lda
from lda_generate_data import LdaDataGenerator
import numpy as np


def main():

    n_topics = 250
    alpha = 0.1
    beta = 0.01    

#     n_docs = 1000
#     vocab_size = 2000
#     document_length = 600
#     gen = LdaDataGenerator(alpha)
#     df = gen.generate_input_df(n_topics, vocab_size, document_length, n_docs)
    
    relative_intensity = True
    fragment_filename = 'input/relative_intensities/Beer_3_T10_POS_fragments_rel.csv'
    neutral_loss_filename = 'input/relative_intensities/Beer_3_T10_POS_losses_rel.csv'
    mzdiff_filename = None    
    ms1_filename = 'input/relative_intensities/Beer_3_T10_POS_ms1_rel.csv'
    ms2_filename = 'input/relative_intensities/Beer_3_T10_POS_ms2_rel.csv'
    ms2lda = Ms2Lda(fragment_filename, neutral_loss_filename, mzdiff_filename, 
                ms1_filename, ms2_filename, relative_intensity)    
    df = ms2lda.preprocess()
    
    start_time = time.time()
    gibbs = CollapseGibbsLda(df, 13, alpha, beta, previous_model=None)
    gibbs.run(n_burn=0, n_samples=3, n_thin=1, use_native=False)
    print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))
        
#     gen._plot_nicely(gibbs.topic_word_, 'Inferred Topics X Terms', 'terms', 'topics')
#     gen._plot_nicely(gibbs.doc_topic_.T, 'Inferred Topics X Docs', 'docs', 'topics')
#     plt.plot(gibbs.all_lls)
#     plt.show()

if __name__ == "__main__":
    main()