from lda_for_fragments import *
    
n_topics = 100
# run_lda('beer3_pos', 'input/Beer_3_T10_POS_combined.csv', False, n_topics)
run_lda('urine37_pos', 'input/Urine_37_Top10_POS_combined.csv', False, n_topics)
