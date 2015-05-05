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

def run_lda(results_prefix, datatype, input_filename, n_topics, n_samples):    
                
    data = pd.read_csv(input_filename, index_col=0)

    # discretise the values by logging and scale it from 0 .. 100
    data = np.log10(data)
    data /= data.max().max()
    data *= 100
        
    # get rid of NaNs, transpose the data and floor it
    data = data.replace(np.nan,0)
    data = data.transpose()
    sd = coo_matrix(data)
    counts, bins, bars = plt.hist(sd.data, bins=range(100))
    plt.title('Discretised intensities')
    plt.xlabel('Bins')
    plt.ylabel('Counts')
    plt.show()
    sd = sd.floor()  
    npdata = np.array(sd.todense(),dtype='int64')
    print "Data shape " + str(npdata.shape)

    print "Fitting model..."
    sys.stdout.flush()
    model = lda.LDA(n_topics = n_topics, n_iter=n_samples, random_state=1)
    model.fit(npdata)
    print "DONE!"
    
    outfile = get_outfile(results_prefix, '_' + datatype + '_topics.csv') 
    print "Writing topics to " + outfile
    topic_fragments = model.topic_word_
    n_top_frags = 20
    with open(outfile,'w') as f:
        for i,topic_dist in enumerate(topic_fragments):
            topic_f = np.array(data.columns.values)[np.argsort(topic_dist)][:-n_top_frags:-1]
            out_string = 'Topic {},{}'.format(i, ','.join(topic_f.astype('str')))
            # print(out_string)
            f.write(out_string+'\n')

    outfile = get_outfile(results_prefix, '_' + datatype + '_all.csv') 
    print "Writing fragments x topics to " + outfile
    topic = model.topic_word_
    masses = np.array(data.transpose().index)
    d = {}
    for i in np.arange(n_topics):
        topic_name = i
        topic_series = pd.Series(topic[i],index=masses)
        d[topic_name] = topic_series
    topicdf = pd.DataFrame(d)
    topicdf.to_csv(outfile)

    outfile = get_outfile(results_prefix, '_' + datatype + '_docs.csv') 
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
    
    return data, model, topicdf, docdf
    
## a lot of similarities, so probably can combine this with the function above .. somehow
def run_lda_combined(results_prefix, fragment_filename, neutral_loss_filename, mzdiff_filename, n_topics, n_samples):    
        
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
    mzdiff_data /= mzdiff_data.max().max()
    mzdiff_data *= 100
    data = data.append(mzdiff_data)
    
    # get rid of NaNs, transpose the data and floor it
    data = data.replace(np.nan,0)
    data = data.transpose()
    sd = coo_matrix(data)
    counts, bins, bars = plt.hist(sd.data, bins=range(100))
    plt.title('Discretised intensities')   
    plt.xlabel('Bins')
    plt.ylabel('Counts')     
    plt.show()
    sd = sd.floor()  
    npdata = np.array(sd.todense(),dtype='int64')
    print "Data shape " + str(npdata.shape)

    print "Fitting model..."
    sys.stdout.flush()
    model = lda.LDA(n_topics = n_topics, n_iter=n_samples, random_state=1)
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
    print "Writing fragments x topics to " + outfile
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
    
    return data, model, topicdf, docdf

# from http://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations
def _get_text_positions(x_data, y_data, txt_width, txt_height):
    a = zip(y_data, x_data)
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height) 
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions

# from http://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations
def _text_plotter(x_data, y_data, text_positions, axis, txt_width, txt_height, fontspec):
    for x,y,t in zip(x_data, y_data, text_positions):
        axis.text(x-txt_width, 1.01*t, '%.5f'%x, rotation=0, **fontspec)
        if y != t:
            axis.arrow(x, t,0,y-t, color='red',alpha=0.3, width=txt_width*0.1, 
                       head_width=txt_width/4, head_length=txt_height*0.5, 
                       zorder=0,length_includes_head=True)
            
def plot_lda_fragments(n_docs, n_fragments, data, model, docdf, ms1, ms2):
    
    topic_fragments = model.topic_word_
    headers = list(docdf.columns.values)

    ms2['bin_id'] = ms2['bin_id'].astype(str)

    for i, topic_dist in enumerate(topic_fragments):

        print "Topic " + str(i)
        print "=========="
        print

        column_values = np.array(docdf.columns.values)    
        doc_dist = docdf.iloc[[i]].as_matrix().flatten()
        idx = np.argsort(doc_dist)[::-1] # argsort in descending order
        topic_d = np.array(column_values)[idx]
        topic_p = np.array(doc_dist)[idx]
        top_n_docs = topic_d[1:n_docs+1]
        top_n_docs_p = topic_p[1:n_docs+1]

        print "Parent peaks"
        print
        print '     %s\t%s\t\t%s\t\t%s\t\t%s' % ('peakID', 'mz', 'rt', 'int', 'prob')
        parent_ids = []
        parent_masses = []
        parent_intensities = []
        parent_all_fragments = {}
        count = 1
        for t in zip(top_n_docs, top_n_docs_p):

            # split mz_rt_peakid string into tokens
            tokens = t[0].split('_')
            peakid = int(tokens[2])
            ms1_row = ms1.loc[[peakid]]
            mz = ms1_row[['mz']]
            mz = np.asscalar(mz.values)
            rt = ms1_row[['rt']]
            rt = np.asscalar(rt.values)
            intensity = ms1_row[['intensity']]
            intensity = np.asscalar(intensity.values)
            prob = t[1]

            print '%-5d%-5d\t%3.5f\t%6.3f\t\t%.3e\t%.3f' % (count, peakid, mz, rt, intensity, prob)
            parent_ids.append(peakid)
            parent_masses.append(mz)
            parent_intensities.append(intensity)
            
            # find all the fragment peaks of this parent peak
            ms2_rows = ms2.loc[ms2['MSnParentPeakID']==peakid]
            peakids = ms2_rows[['peakID']]
            mzs = ms2_rows[['mz']]
            intensities = ms2_rows[['intensity']]
            parentids = ms2_rows[['MSnParentPeakID']]

            # convert from pandas dataframes to list
            peakids = peakids.values.ravel().tolist()
            mzs = mzs.values.ravel().tolist()
            intensities = intensities.values.ravel().tolist()
            parentids = parentids.values.ravel().tolist()

            # save all the fragment peaks of this parent peak into the dictionary
            parentid = peakid
            items = []
            for n in range(len(peakids)):
                mz = mzs[n]
                intensity = intensities[n]
                fragment_peakid = peakids[n]
                item = (fragment_peakid, parentid, mz, intensity)
                items.append(item)
            parent_all_fragments[parentid] = items

            count += 1

        min_parent_mz = np.min(np.array(parent_masses))
        max_parent_mz = np.max(np.array(parent_masses))
        column_values = np.array(data.columns.values)
        idx = np.argsort(topic_dist)[::-1] # argsort in descending order
        topic_w = np.array(column_values)[idx]
        topic_p = np.array(topic_dist)[idx]    
        fragments = []
        fragments_p = []
        for w, p in zip(topic_w, topic_p):
            if len(fragments) >= n_fragments:
                break
            fragments.append(w)
            fragments_p.append(p)

        print
        print "Fragments"
        print
        parent_topic_fragments = {}
        count = 1
        for t in zip(fragments, fragments_p):

            fragment = t[0]
            tokens = fragment.split('_')
            bin_id = tokens[1]
            bin_prob = t[1]
            ms2_rows = ms2.loc[ms2['bin_id']==bin_id]
            ms2_rows = ms2_rows.loc[ms2_rows['MSnParentPeakID'].isin(parent_ids)]

            print '%-5d%s (%.3f)' % (count, t[0], t[1])
            print ms2_rows[['peakID', 'MSnParentPeakID', 'mz', 'rt', 'intensity']].to_string(index=False, justify='left')

            count += 1

            peakids = ms2_rows[['peakID']]
            mzs = ms2_rows[['mz']]
            intensities = ms2_rows[['intensity']]
            parentids = ms2_rows[['MSnParentPeakID']]

            # convert from pandas dataframes to list
            peakids = peakids.values.ravel().tolist()
            mzs = mzs.values.ravel().tolist()
            intensities = intensities.values.ravel().tolist()
            parentids = parentids.values.ravel().tolist()

            for n in range(len(parentids)):
                parentid = parentids[n]
                mz = mzs[n]
                intensity = intensities[n]
                peakid = peakids[n]
                item = (peakid, parentid, mz, intensity)
                if parentid in parent_topic_fragments:
                    existing_list = parent_topic_fragments[parentid]
                    existing_list.append(item)
                else:
                    new_list = [item]
                    parent_topic_fragments[parentid] = new_list

        print
        sys.stdout.flush()

        # plot the n_docs parent peaks in this topic
        parent_fontspec = {
            'size':'10', 
            'color':'blue', 
            'weight':'bold'
        }
        fragment_fontspec = {
            'size':'8', 
            'color':'black', 
            'weight':'bold'
        }
        
        # make plot for every parent peak
        num_peaks = len(parent_ids)
        for n in range(num_peaks):

            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            #set the bbox for the text. Increase txt_width for wider text.
            txt_width = 30*(plt.xlim()[1] - plt.xlim()[0])
            txt_height = 1*(plt.ylim()[1] - plt.ylim()[0])

            # plot the parent peak first
            parent_mass = parent_masses[n]
            parent_intensity = np.log10(parent_intensities[n])
            plt.plot((parent_mass, parent_mass), (0, parent_intensity), linewidth=2.0, color='b')
            x = parent_mass
            y = parent_intensity
            parent_id = parent_ids[n]
            label = "%.5f" % parent_mass
            plt.text(x, y, label, **parent_fontspec)   

            # plot all the fragment peaks of this parent peak
            fragments_list = parent_all_fragments[parent_id]
            num_peaks = len(fragments_list)
            for j in range(num_peaks):
                item = fragments_list[j]
                peakid = item[0]
                parentid = item[1]
                mass = item[2]
                intensity = np.log10(item[3])
                plt.plot((mass, mass), (0, intensity), linewidth=1.0, color='#FF9933')

            # plot the fragment peaks in this topic that also occur in this parent peak
            if parent_id in parent_topic_fragments:        
                fragments_list = parent_topic_fragments[parent_id]
                num_peaks = len(fragments_list)
                x_data = []
                y_data = []
                for j in range(num_peaks):
                    item = fragments_list[j]
                    peakid = item[0]
                    parentid = item[1]
                    mass = item[2]
                    intensity = np.log10(item[3])
                    plt.plot((mass, mass), (0, intensity), linewidth=2.0, color='#800000')
                    x = mass
                    y = intensity
                    x_data.append(x)
                    y_data.append(y)
                
                # Get the corrected text positions, then write the text.
                x_data = np.array(x_data)
                y_data = np.array(y_data)
                text_positions = _get_text_positions(x_data, y_data, txt_width, txt_height)
                _text_plotter(x_data, y_data, text_positions, ax, txt_width, txt_height, fragment_fontspec)

            xlim_upper = max_parent_mz + 100
            plt.xlim([0, xlim_upper])
            plt.ylim([0, 15])
            plt.xlabel('m/z')
            plt.ylabel('log10(intensity)')
            plt.title('Topic ' + str(i) + ' -- parent peak ' + ("%.5f" % parent_mass))
            plt.show()
            
        # break