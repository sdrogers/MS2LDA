from lda_for_fragments import run_lda

n_topics = 100
n_samples = 500

# run_lda('beer3_pos', 'input/Beer_3_T10_POS_fragments.csv', 'input/Beer_3_T10_POS_losses.csv', 'input/Beer_3_T10_POS_mzdiffs.csv', n_topics, n_samples)

# run_lda('beer3_neg', 'input/Beer_3_T10_NEG_fragments.csv', 'input/Beer_3_T10_NEG_losses.csv', 'input/Beer_3_T10_NEG_mzdiffs.csv', n_topics, n_samples)

run_lda('urine37_pos', 'input/Urine_37_Top10_POS_fragments.csv', 'input/Urine_37_Top10_POS_losses.csv', 'input/Urine_37_Top10_POS_mzdiffs.csv', n_topics, n_samples)

run_lda('urine37_neg', 'input/Urine_37_Top10_NEG_fragments.csv', 'input/Urine_37_Top10_NEG_losses.csv', 'input/Urine_37_Top10_NEG_mzdiffs.csv', n_topics, n_samples)

