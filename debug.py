from lda_for_fragments import Ms2Lda

ms2lda = Ms2Lda.resume_from('notebooks/results/beerQCNEG_test13August.project')
ms2lda.do_thresholding(th_doc_topic=0, th_topic_word=0.05)
ms2lda.print_topic_words()

# topic_sort_criteria, sorted_topic_counts = ms2lda.rank_topics(sort_by="h_index")
# ms2lda.plot_lda_fragments(consistency=0.33, sort_by="h_index", interactive=True)