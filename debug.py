from lda_for_fragments import Ms2Lda

ms2lda = Ms2Lda.resume_from('notebooks/results/beer3pos.project')
ms2lda.do_thresholding(th_doc_topic=0.05, th_topic_word=0.01)
ms2lda.print_topic_words()

special_nodes = [
    ('doc_21213', '#CC0000'), # maroon
    ('doc_21758', 'gold'),
    ('doc_21182', 'green'),
    ('topic_240', '#CC0000'), # maroon
    ('topic_76', 'aqua'),
    ('topic_253', '#ff1493') # deep pink
]
ms2lda.plot_lda_fragments(consistency=0.0, interactive=True, to_highlight=special_nodes)