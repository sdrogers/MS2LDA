import numpy as np
import pandas as pd
import lda
import lda.datasets
import pylab as plt
from scipy.sparse import coo_matrix

import sys
import os

def get_outfile(results_prefix, doctype):
    parent_dir = 'results/' + results_prefix
    outfile = parent_dir + '/' + results_prefix + doctype
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)        
    return outfile
    
def run_lda(results_prefix, fragment_filename, neutral_loss_filename, mzdiff_filename, n_topics):    
        
    fragment_data = pd.read_csv(fragment_filename, index_col=0)
    neutral_loss_data = pd.read_csv(neutral_loss_filename, index_col=0)
    mzdiff_data = pd.read_csv(mzdiff_filename, index_col=0)

    # discretise the fragment and neutral loss intensities values
    # log and scale it from 0 .. 100
    data = fragment_data.append(neutral_loss_data)
    data = np.log10(data)
    data /= data.max().max()
    data *= 100
    
    # then scale mzdiff counts from 0 .. 100 too, and append it to data
    mzdiff_data = np.log10(mzdiff_data)
    mzdiff_data /= mzdiff_data.max().max()
    mzdiff_data *= 100    
    data = data.append(mzdiff_data)
    
    # get rid of NaNs, transpose the data and floor it
    data = data.replace(np.nan,0)
    data = data.transpose()
    sd = coo_matrix(data)
    plt.hist(sd.data)
    plt.show()
    sd = sd.floor()  
    npdata = np.array(sd.todense(),dtype='int64')
    print "Data shape " + str(npdata.shape)

    print "Fitting model..."
    sys.stdout.flush()
    model = lda.LDA(n_topics = n_topics, n_iter=500, random_state=1)
    model.fit(npdata)
    print "DONE!"
    
    outfile = get_outfile(results_prefix, '_topics.csv') 
    print "Writing topics to " + outfile
    topic_fragments = model.topic_word_
    n_top_frags = 20
    with open(outfile,'w') as f:
        for i,topic_dist in enumerate(topic_fragments):
            topic_f = np.array(data.columns.values)[np.argsort(topic_dist)][:-n_top_frags:-1]
            out_string = 'Topic {},{}'.format(i, ','.join(topic_f.astype('str')))
            # print(out_string)
            f.write(out_string+'\n')

    outfile = get_outfile(results_prefix, '_all.csv') 
    print "Writing fragments x topics probability matrix to " + outfile
    topic = model.topic_word_
    masses = np.array(data.transpose().index)
    d = {}
    for i in np.arange(n_topics):
        topic_name = i
        topic_series = pd.Series(topic[i],index=masses)
        d[topic_name] = topic_series
    topicdf = pd.DataFrame(d)
    topicdf.to_csv(outfile)

    outfile = get_outfile(results_prefix, '_docs.csv') 
    print "Writing topic docs to " + outfile
    doc = model.doc_topic_
    (n_doc, a) = doc.shape
    topic_index = np.arange(n_topics)
    doc_names = np.array(data.index)
    d = {}
    for i in np.arange(n_doc):
        doc_name = doc_names[i]
        doc_series = pd.Series(doc[i],index=topic_index)
        d[doc_name] = doc_series
    docdf = pd.DataFrame(d)
    cols = docdf.columns.tolist()
    mass_rt = [(float(m.split('_')[0]),float(m.split('_')[1])) for m in cols]
    sorted_mass_rt = sorted(mass_rt,key=lambda m:m[0])
    ind = [mass_rt.index(i) for i in sorted_mass_rt]
    docdf = docdf[ind]
    docdf.to_csv(outfile)
