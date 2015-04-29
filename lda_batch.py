from lda_for_fragments import run_lda

n_topics = 100
run_lda('beer3_pos', 'input/Beer_3_T10_POS_fragments.csv', 'input/Beer_3_T10_POS_losses.csv', 
        'input/Beer_3_T10_POS_mzdiff.csv', n_topics)