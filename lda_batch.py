from lda_for_fragments import *
    
n_topics = 50

# beer positive mode
# run_lda('beer_pos', 'input/QC_Beer_ParentsFragments_Tonyscripting.csv', True, n_topics)
# run_lda('beer_pos', 'input/QC_Beer_ParentsFragments_Tonyscripting.csv', False, n_topics)

# beer negative mode
# run_lda('beer_neg', 'input/QC_Beer_negmode_7ppm.csv', True, n_topics)
# run_lda('beer_neg', 'input/QC_Beer_negmode_7ppm.csv', False, n_topics)

# Two matrices of a beer sample from the 'new data' obtained this week ... This one is the 7 Giraffes Extraordinary Ale ...
n_topics = 100
run_lda('beer3_pos', 'input/FP_Matrix_Beer_3_POS_7ppm_test_IDEOMsettings_SpecPeaks_combined.csv', False, n_topics)

# urine37 mixed modes
# run_lda('urine37_mixed', 'input/Urine_FragmentsParents_MS2filter5000_Tonyscripting.csv', True, n_topics)
# run_lda('urine37_mixed', 'input/Urine_FragmentsParents_MS2filter5000_Tonyscripting.csv', False, n_topics)

# urine37 positive mode
# run_lda('urine37_pos', 'input/Urine37_mixed_posmode_7ppm.csv', True, n_topics)
# run_lda('urine37_pos', 'input/Urine37_mixed_posmode_7ppm.csv', False, n_topics)

# urine37 negative mode
# run_lda('urine37_neg', 'input/Urine37_mixed_negmode_7ppm.csv', True, n_topics)
# run_lda('urine37_neg', 'input/Urine37_mixed_negmode_7ppm.csv', False, n_topics)

# campylobacter positive mode
# run_lda('campy_pos', 'input/CampyT10_pos.csv', True, n_topics)
# run_lda('campy_pos', 'input/CampyT10_pos.csv', False, n_topics)

# campylobacter negative mode
# run_lda('campy_neg', 'input/CampyT10_neg.csv', True, n_topics)
# run_lda('campy_neg', 'input/CampyT10_neg.csv', False, n_topics)

# urine94 mixed modes
# run_lda('urine94_mixed', 'input/ParentFragmentMatrix_Urine94_mixed.csv', True, n_topics)
# run_lda('urine94_mixed', 'input/ParentFragmentMatrix_Urine94_mixed.csv', False, n_topics)

# urine94 positive mode
# run_lda('urine94_pos', 'input/Urine94_mixed_posmode_7ppm.csv', True, n_topics)
# run_lda('urine94_pos', 'input/Urine94_mixed_posmode_7ppm.csv', False, n_topics)

# urine94 negative mode
# run_lda('urine94_neg', 'input/Urine94_mixed_negmode_7ppm.csv', True, n_topics)
# run_lda('urine94_neg', 'input/Urine94_mixed_negmode_7ppm.csv', False, n_topics)

# some ecoli thing?
# run_lda('Ecoli_APEC_pos', 'input/Ecoli_APEC_WC_T10_pos.csv', True, n_topics)
# run_lda('Ecoli_APEC_pos', 'input/Ecoli_APEC_WC_T10_pos.csv', False, n_topics)

# some ecoli thing?
# run_lda('Ecoli_NISSLE_pos', 'input/Ecoli_NISSLE_WC_T10_pos.csv', True, n_topics)
# run_lda('Ecoli_NISSLE_pos', 'input/Ecoli_NISSLE_WC_T10_pos.csv', False, n_topics)
