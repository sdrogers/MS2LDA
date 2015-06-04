import operator
import os
import re
import sys
import time
import timeit

from numpy.random.mtrand import RandomState
from pandas.core.frame import DataFrame
from scipy.sparse import coo_matrix

from lda_cgs import CollapseGibbsLda
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import pylab as plt


try:
    from lda import LDA
except Exception:
    pass



class Ms2Lda:
    
    def __init__(self, fragment_filename, neutral_loss_filename, mzdiff_filename, 
                 ms1_filename, ms2_filename, relative_intensity=False):
        
        self.fragment_data = pd.read_csv(fragment_filename, index_col=0)
        self.neutral_loss_data = pd.read_csv(neutral_loss_filename, index_col=0)
        if mzdiff_filename is not None:
            self.mzdiff_data = pd.read_csv(mzdiff_filename, index_col=0)
        else:
            self.mzdiff_data = None
        self.ms1 = pd.read_csv(ms1_filename, index_col=0)
        self.ms2 = pd.read_csv(ms2_filename, index_col=0)
            
        self.relative_intensity = relative_intensity
        self.EPSILON = 0.05
        
    def preprocess(self):

        self.ms2['fragment_bin_id'] = self.ms2['fragment_bin_id'].astype(str)
        self.ms2['loss_bin_id'] = self.ms2['loss_bin_id'].astype(str)

        if self.relative_intensity: 

            # discretise the fragment and neutral loss intensities values
            # values are already normalised from 0 .. 1 during feature extraction
            self.data = self.fragment_data.append(self.neutral_loss_data)
            self.data *= 100 # so just convert to 0 .. 100
        
            # then scale mzdiff counts from 0 .. 100 too, and append it to data    
            if self.mzdiff_data is not None:
                self.mzdiff_data *= 100
                self.data = self.data.append(self.mzdiff_data)
        
        else: # absolute intensity values

            # discretise the fragment and neutral loss intensities values
            # log and scale it from 0 .. 100
            self.data = self.fragment_data.append(self.neutral_loss_data)
            self.data = np.log10(self.data)
            self.data /= self.data.max().max()
            self.data *= 100
        
            # then scale mzdiff counts from 0 .. 100 too, and append it to data    
            if self.mzdiff_data is not None:
                self.mzdiff_data /= self.mzdiff_data.max().max()
                self.mzdiff_data *= 100
                self.data = self.data.append(self.mzdiff_data)
        
        # get rid of NaNs, transpose the data and floor it
        self.data = self.data.replace(np.nan,0)
        self.data = self.data.transpose()
        vocab = self.data.columns.values
        
        sd = coo_matrix(self.data)
#         counts, bins, bars = plt.hist(sd.data, bins=range(100))
#         plt.title('Discretised intensities')   
#         plt.xlabel('Bins')
#         plt.ylabel('Counts')     
#         plt.show()
        sd = sd.floor()  
        npdata = np.array(sd.todense(),dtype='int64')

        print "Data shape " + str(npdata.shape)
        df = DataFrame(npdata)

        return df, vocab

    def run_lda(self, df, vocab, n_topics, n_samples, n_burn, n_thin, alpha, beta, 
                use_own_model=False, use_native=False, previous_model=None):    
                        
        print "Fitting model..."
        self.n_topics = n_topics
        rng = RandomState(1234567890)
        sys.stdout.flush()
        if use_own_model:
            self.model = CollapseGibbsLda(df, vocab, n_topics, alpha, beta, previous_model=previous_model)
            self.n_topics = self.model.K # might change if previous_model is used
            start = timeit.default_timer()
            self.model.run(n_burn, n_samples, n_thin, use_native=use_native)
        else:
            self.model = LDA(n_topics=n_topics, n_iter=n_samples, random_state=rng, alpha=alpha, eta=beta)
            start = timeit.default_timer()
            self.model.fit(df.as_matrix())
        stop = timeit.default_timer()
        print "DONE. Time=" + str(stop-start)
        
    def _natural_sort(self, l): 
        convert = lambda text: int(text) if text.isdigit() else text.lower() 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(l, key = alphanum_key)
                
    def write_results(self, results_prefix):

        previous_model = self.model.previous_model
        selected_topics = None
        if previous_model is not None and hasattr(previous_model, 'selected_topics'):
            selected_topics = previous_model.selected_topics
        
        # create topic-word output file
        # the column names of each topic is assigned here
        topic_names = []
        outfile = self._get_outfile(results_prefix, '_topics.csv') 
        print "Writing topics to " + outfile
        topic_fragments = self.model.topic_word_
        with open(outfile,'w') as f:
            
            counter = 0
            for i, topic_dist in enumerate(topic_fragments):

#                 topic_f = np.array(self.data.columns.values)[np.argsort(topic_dist)][:-n_top_frags:-1]
#                 out_string = 'Topic {},{}'.format(i, ','.join(topic_f.astype('str')))
#                 f.write(out_string+'\n')

                ordering = np.argsort(topic_dist)
                vocab = self.data.columns.values                
                topic_words = np.array(vocab)[ordering][::-1]
                dist = topic_dist[ordering][::-1]
                
                if selected_topics is not None:
                    if i < len(selected_topics):
                        topic_name = 'Fixed Topic {}'.format(selected_topics[i])
                    else:
                        topic_name = 'Topic {}'.format(counter)
                        counter += 1
                else:
                    topic_name = 'Topic {}'.format(i)                    
                f.write(topic_name)
                topic_names.append(topic_name)
                
                # filter entries to display by epsilon
                for j in range(len(topic_words)):
                    if dist[j] > self.EPSILON:
                        f.write(',{}'.format(topic_words[j]))
                    else:
                        break
                f.write('\n')
    
        outfile = self._get_outfile(results_prefix, '_all.csv') 
        print "Writing fragments x topics to " + outfile

        # create document-topic output file        
        topic = self.model.topic_word_
        masses = np.array(self.data.transpose().index)
        d = {}
        for i in np.arange(self.n_topics):
            topic_name = topic_names[i]
            topic_series = pd.Series(topic[i], index=masses)
            d[topic_name] = topic_series
        self.topicdf = pd.DataFrame(d)
        
        # make sure that columns in topicdf are in the correct order
        # because many times we'd index the columns in the dataframes directly by their positions
        cols = self.topicdf.columns.tolist()
        sorted_cols = self._natural_sort(cols)
        self.topicdf = self.topicdf[sorted_cols]        

        # threshold topicdf to get rid of small values
        def f(x):
            if x < self.EPSILON: return 0
            else: return x
        self.topicdf = self.topicdf.applymap(f)
        self.topicdf.to_csv(outfile)
    
        # create topic-docs output file
        doc = self.model.doc_topic_
        (n_doc, a) = doc.shape
        topic_index = np.arange(self.n_topics)
        doc_names = np.array(self.data.index)
        d = {}
        for i in np.arange(n_doc):
            doc_name = doc_names[i]
            doc_series = pd.Series(doc[i], index=topic_index)
            d[doc_name] = doc_series
        self.docdf = pd.DataFrame(d)
        
        # sort columns by mass_rt values
        cols = self.docdf.columns.tolist()
        mass_rt = [(float(m.split('_')[0]),float(m.split('_')[1])) for m in cols]
        sorted_mass_rt = sorted(mass_rt,key=lambda m:m[0])
        ind = [mass_rt.index(i) for i in sorted_mass_rt]
        self.docdf = self.docdf[ind]
        # self.docdf.to_csv(outfile)

        # threshold docdf to get rid of small values and also scale it
        self.docdf = self.docdf.applymap(f)                
        for i, row in self.docdf.iterrows(): # iterate through the rows
            doc = self.docdf.ix[:, i]
            selected = doc[doc>0]
            count = len(selected.values)
            selected = selected * count
            self.docdf.ix[:, i] = selected
        self.docdf = self.docdf.replace(np.nan, 0)
        outfile = self._get_outfile(results_prefix, '_docs.csv') 
        print "Writing topic docs to " + outfile
        self.docdf.transpose().to_csv(outfile)
        
    def save_model(self, topic_indices, model_out, words_out):
        self.model.save(topic_indices, model_out, words_out)
        
    def rank_topics(self, sort_by="h_index", selected_topics=None, top_N=None):

        if sort_by == 'h_index':
            topic_ranking = self._h_index() 
        elif sort_by == 'in_degree':
            topic_ranking = self._in_degree() 
        
        sorted_topic_counts = sorted(topic_ranking.items(), key=operator.itemgetter(1), reverse=True)
        print "Topic Ranking"
        print "============="
        print
        counter = 0
        for topic_number, score in sorted_topic_counts:
            if selected_topics is not None:
                if topic_number not in selected_topics:
                    continue
            if sort_by == 'h_index':
                print "Topic " + str(topic_number) + " h-index=" + str(score)
            elif sort_by == 'in_degree':
                print "Topic " + str(topic_number) + " in-degree=" + str(score)
            counter += 1
            if top_N is not None and counter >= top_N:
                break
        print
        
        return topic_ranking, sorted_topic_counts
                
    def plot_lda_fragments(self, consistency=0.50, sort_by="h_index", selected_topics=None):
                
        topic_ranking, sorted_topic_counts = self.rank_topics(sort_by=sort_by, selected_topics=selected_topics)        
        for (i, c) in sorted_topic_counts:
            
            # skip non-selected topics
            if selected_topics is not None:
                if i not in selected_topics:
                    continue

            if sort_by == 'h_index':
                print "Topic " + str(i) + " h-index=" + str(topic_ranking[i])
            elif sort_by == 'in_degree':
                print "Topic " + str(i) + " in-degree=" + str(topic_ranking[i])
            print "====================="
            print

            column_values = np.array(self.docdf.columns.values)    
            doc_dist = self.docdf.iloc[[i]].as_matrix().flatten()
            
            # argsort in descending order
            idx = np.argsort(doc_dist)[::-1] 
            topic_d = np.array(column_values)[idx]
            topic_p = np.array(doc_dist)[idx]
            
            # pick the top-n documents
            # top_n_docs = topic_d[1:n_docs+1]
            # top_n_docs_p = topic_p[1:n_docs+1]

            # pick the documents with non-zero values
            nnz_idx = topic_p>0
            top_n_docs = topic_d[nnz_idx]
            top_n_docs_p = topic_p[nnz_idx]
            
            if len(top_n_docs) == 0:
                print "No parent peaks above the threshold found for this topic"
                continue
    
            print "Parent peaks"
            print
            print '     %s\t%s\t\t%s\t\t%s\t\t%s' % ('peakID', 'mz', 'rt', 'int', 'score')
            parent_ids = []
            parent_masses = []
            parent_intensities = []
            parent_all_fragments = {}
            count = 1
            for t in zip(top_n_docs, top_n_docs_p):
    
                # split mz_rt_peakid string into tokens
                tokens = t[0].split('_')
                peakid = int(tokens[2])
                ms1_row = self.ms1.loc[[peakid]]
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
                ms2_rows = self.ms2.loc[self.ms2['MSnParentPeakID']==peakid]
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
    
            sys.stdout.flush()
            max_parent_mz = np.max(np.array(parent_masses))

            # argsort in descending order by p(w|d)
            word_dist = self.topicdf.transpose().iloc[[i]].as_matrix().flatten()                          
            column_values = np.array(self.topicdf.transpose().columns.values)    

            # argsort in descending order
            idx = np.argsort(word_dist)[::-1] 
            topic_w = np.array(column_values)[idx]
            topic_p = np.array(word_dist)[idx]    
            
            # pick the words with non-zero values
            nnz_idx = topic_p>0
            topic_w = topic_w[nnz_idx]
            topic_p = topic_p[nnz_idx]
            
            # split words into either fragment or loss words                        
            fragments = []
            fragments_p = []
            losses = []
            losses_p = []
            counter = 0
            for w, p in zip(topic_w, topic_p):
                if w.startswith('fragment'):
                    fragments.append(w)
                    fragments_p.append(p)
                elif w.startswith('loss'):
                    losses.append(w)
                    losses_p.append(p)
                counter += 1

            wordfreq = {}
                    
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
                ms2_rows = self.ms2.loc[self.ms2['fragment_bin_id']==bin_id]
                ms2_rows = ms2_rows.loc[ms2_rows['MSnParentPeakID'].isin(parent_ids)]

                print '%-5d%s (%.3f)' % (count, t[0], t[1])
                if not ms2_rows.empty:
                    print ms2_rows[['peakID', 'MSnParentPeakID', 'mz', 'rt', 'intensity']].to_string(index=False, justify='left')
                else:
                    print "\tNothing found for the selected parent peaks"
    
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
                    word = fragment
                    item = (peakid, parentid, mz, intensity, word)
                    if parentid in parent_topic_fragments:
                        existing_list = parent_topic_fragments[parentid]
                        existing_list.append(item)
                    else:
                        new_list = [item]
                        parent_topic_fragments[parentid] = new_list
                    # count how many times this fragment word appears in the retrieved set
                    if fragment in wordfreq:
                        wordfreq[fragment] = wordfreq[fragment] + 1
                    else:
                        wordfreq[fragment] = 1

            print
            print "Losses"
            print
            parent_topic_losses = {}
            count = 1
            for t in zip(losses, losses_p):
    
                loss = t[0]
                tokens = loss.split('_')
                bin_id = tokens[1]
                bin_prob = t[1]
                ms2_rows = self.ms2.loc[self.ms2['loss_bin_id']==bin_id]
                ms2_rows = ms2_rows.loc[ms2_rows['MSnParentPeakID'].isin(parent_ids)]

                print '%-5d%s (%.3f)' % (count, t[0], t[1])
                if not ms2_rows.empty:
                    print ms2_rows[['peakID', 'MSnParentPeakID', 'mz', 'rt', 'intensity']].to_string(index=False, justify='left')
                else:
                    print "\tNothing found for the selected parent peaks"

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
                    word = loss
                    item = (peakid, parentid, mz, intensity, word)
                    if parentid in parent_topic_losses:
                        existing_list = parent_topic_losses[parentid]
                        existing_list.append(item)
                    else:
                        new_list = [item]
                        parent_topic_losses[parentid] = new_list
                    # count how many times this fragment word appears in the retrieved set
                    if loss in wordfreq:
                        wordfreq[loss] = wordfreq[loss] + 1
                    else:
                        wordfreq[loss] = 1
    
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
                'color':'#800000', 
                'weight':'bold'
            }
            loss_fontspec = {
                'size':'8', 
                'color':'green', 
                'weight':'bold'
            }
            
            # make plot for every parent peak
            num_peaks = len(parent_ids)
            for n in range(num_peaks):
    
                figsize=(10, 6)
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
                
                #set the bbox for the text. Increase txt_width for wider text.
                txt_width = 20*(plt.xlim()[1] - plt.xlim()[0])
                txt_height = 0.2*(plt.ylim()[1] - plt.ylim()[0])
    
                # plot the parent peak first
                parent_mass = parent_masses[n]
                if self.relative_intensity:
                    parent_intensity = 0.25
                else:
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
                    if self.relative_intensity:
                        intensity = item[3]
                    else:
                        intensity = np.log10(item[3])
                    plt.plot((mass, mass), (0, intensity), linewidth=1.0, color='#FF9933')

                x_data = []
                y_data = []
                line_type = []
    
                # plot the fragment peaks in this topic that also occur in this parent peak
                if parent_id in parent_topic_fragments:        
                    fragments_list = parent_topic_fragments[parent_id]
                    num_peaks = len(fragments_list)
                    for j in range(num_peaks):
                        item = fragments_list[j]
                        peakid = item[0]
                        parentid = item[1]
                        mass = item[2]
                        if self.relative_intensity:
                            intensity = item[3]
                        else:
                            intensity = np.log10(item[3])
                        word = item[4]
                        freq = wordfreq[word]
                        ratio = float(freq)/len(parent_ids)
                        if ratio >= consistency:
                            plt.plot((mass, mass), (0, intensity), linewidth=2.0, color='#800000')
                            x = mass
                            y = intensity
                            x_data.append(x)
                            y_data.append(y)
                            line_type.append('fragment')
                    
                # plot the neutral losses in this topic that also occur in this parent peak
                if parent_id in parent_topic_losses:        
                    losses_list = parent_topic_losses[parent_id]
                    num_peaks = len(losses_list)
                    for j in range(num_peaks):
                        item = losses_list[j]
                        peakid = item[0]
                        parentid = item[1]
                        mass = item[2]
                        if self.relative_intensity:
                            intensity = item[3]
                        else:
                            intensity = np.log10(item[3])
                        word = item[4]
                        freq = wordfreq[word]
                        ratio = float(freq)/len(parent_ids)
                        if ratio >= consistency:
                            plt.plot((mass, mass), (0, intensity), linewidth=2.0, color='green')
                            x = mass
                            y = intensity
                            x_data.append(x)
                            y_data.append(y)
                            line_type.append('loss')
                    
                # Get the corrected text positions, then write the text.
                x_data = np.array(x_data)
                y_data = np.array(y_data)
                text_positions = self._get_text_positions(x_data, y_data, txt_width, txt_height)
                self._text_plotter(x_data, y_data, line_type, text_positions, ax, txt_width, txt_height, 
                                   fragment_fontspec, loss_fontspec)
    
                xlim_upper = max_parent_mz + 100
                plt.xlim([0, xlim_upper])
                plt.ylim([0, 1.5])

                plt.xlabel('m/z')
                if self.relative_intensity:
                    plt.ylabel('relative intensity')                    
                else:
                    plt.ylabel('log10(intensity)')
                plt.title('Topic ' + str(i) + ' -- parent peak ' + ("%.5f" % parent_mass))
                
                blue_patch = mpatches.Patch(color='blue', label='Parent peak')
                yellow_patch = mpatches.Patch(color='#FF9933', label='Fragment peaks')
                red_patch = mpatches.Patch(color='#800000', label='Topic fragment')
                green_patch = mpatches.Patch(color='green', label='Topic loss')                
                plt.legend(handles=[blue_patch, yellow_patch, red_patch, green_patch])
                
                plt.show()
            
            # break        
            
    def _get_outfile(self, results_prefix, doctype):
        parent_dir = 'results/' + results_prefix
        outfile = parent_dir + '/' + results_prefix + doctype
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)        
        return outfile
        
    # from http://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations
    def _get_text_positions(self, x_data, y_data, txt_width, txt_height):
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
    def _text_plotter(self, x_data, y_data, line_type, text_positions, axis, txt_width, txt_height, 
                      fragment_fontspec, loss_fontspec):
        for x,y,t,l in zip(x_data, y_data, text_positions, line_type):
            if l == 'fragment':
                axis.text(x-txt_width, 1.01*t, '%.5f'%x, rotation=0, **fragment_fontspec)
            elif l == 'loss':
                axis.text(x-txt_width, 1.01*t, '%.5f'%x, rotation=0, **loss_fontspec)                
            if y != t:
                axis.arrow(x, t,0,y-t, color='black', alpha=0.2, width=txt_width*0.01, 
                           head_width=txt_width/4, head_length=txt_height*0.25, 
                           zorder=0,length_includes_head=True)
    
    # compute the h-index of topics
    def _h_index(self):
                
        topic_fragments = self.model.topic_word_
        topic_counts = {}

        for i, topic_dist in enumerate(topic_fragments):
            
            sys.stdout.flush()
            
            # find the words in this topic above the threshold
            topic_words = self.topicdf.ix[:, i]
            topic_words = topic_words.iloc[topic_words.nonzero()[0]]      

            fragment_words = {}
            loss_words = {}            
            for word in topic_words.index:
                tokens = word.split('_')
                word_type = tokens[0]
                value = tokens[1]
                if word_type == 'fragment':
                    fragment_words[value] = 0
                elif word_type == 'loss':
                    loss_words[value] = 0
            
            # find the documents in this topic above the threshold
            topic_docs = self.docdf.ix[i, :]
            topic_docs = topic_docs.iloc[topic_docs.nonzero()[0]]
            
            # handle empty topics
            if topic_docs.empty:
                
                topic_counts[i] = 0
                
            else:
            
                # now find out how many of the documents in this topic actually 'cite' the words    
                for docname in topic_docs.index:
    
                    # split mz_rt_peakid string into tokens
                    tokens = docname.split('_')
                    peakid = int(tokens[2])
                    
                    # find all the fragment peaks of this parent peak
                    ms2_rows = self.ms2.loc[self.ms2['MSnParentPeakID']==peakid]
                    fragment_bin_ids = ms2_rows[['fragment_bin_id']]
                    loss_bin_ids = ms2_rows[['loss_bin_id']]       
                    
                    # convert from pandas dataframes to list
                    fragment_bin_ids = fragment_bin_ids.values.ravel().tolist()
                    loss_bin_ids = loss_bin_ids.values.ravel().tolist()
                    
                    # count the citation numbers
                    for cited in fragment_bin_ids:
                        if cited == 'nan':
                            continue
                        else:
                            if cited in fragment_words:
                                fragment_words[cited] = fragment_words[cited] + 1
                    for cited in loss_bin_ids:
                        if cited == 'nan':
                            continue
                        else:
                            if cited in loss_words:
                                loss_words[cited] = loss_words[cited] + 1
                    
                    # make a dataframe of the articles & citation counts
                    fragment_df = DataFrame(fragment_words, index=['counts']).transpose()
                    loss_df = DataFrame(loss_words, index=['counts']).transpose()
                    df = fragment_df.append(loss_df)
                    df = df.sort(['counts'], ascending=False)
                    
                    # compute the h-index
                    h_index = 0
                    for index, row in df.iterrows():
                        if row['counts'] > h_index:
                            h_index += 1
                        else:
                            break
    
                topic_counts[i] = h_index
            
        return topic_counts
    
    def _get_docname(self, row_index):
        tokens = row_index.split('_')
        docname = "doc_"+ str(tokens[0]) + "_" + str(tokens[1])
        return docname
    
    def _get_topicname(self, col_index):
        topic = "topic_" + str(col_index)
        return topic    
    
    # computes the in-degree of topics
    def _in_degree(self):    

        topic_fragments = self.model.topic_word_
        topic_counts = {}
        
        for i, topic_dist in enumerate(topic_fragments):
            
            sys.stdout.flush()
            
            # find the documents in this topic above the threshold
            topic_docs = self.docdf.ix[i, :]
            topic_docs = topic_docs.iloc[topic_docs.nonzero()[0]]
            topic_counts[i] = len(topic_docs)
            
        return topic_counts                
            
def main():
        
    if len(sys.argv)>1:
        n_topics = int(sys.argv[1])
    else:
        n_topics = 125
    n_samples = 10
    n_burn = 0
    n_thin = 1
    alpha = 0.1
    beta = 0.01

    # train on beer3pos
    
    relative_intensity = True
    fragment_filename = 'input/relative_intensities/Beer_3_T10_POS_fragments_rel.csv'
    neutral_loss_filename = 'input/relative_intensities/Beer_3_T10_POS_losses_rel.csv'
    mzdiff_filename = None    
    ms1_filename = 'input/relative_intensities/Beer_3_T10_POS_ms1_rel.csv'
    ms2_filename = 'input/relative_intensities/Beer_3_T10_POS_ms2_rel.csv'
 
    ms2lda = Ms2Lda(fragment_filename, neutral_loss_filename, mzdiff_filename, 
                ms1_filename, ms2_filename, relative_intensity)    
    df, vocab = ms2lda.preprocess()    
    ms2lda.run_lda(df, vocab, n_topics, n_samples, n_burn, n_thin, 
                   alpha, beta, use_own_model=True, use_native=True)
    ms2lda.write_results('beer3pos')

    # save some topics from beer3pos lda
    
    selected_topics = [0, 1, 2, 3, 4, 5]    
    ms2lda.save_model(selected_topics, 'input/beer3pos.model.p', 'input/beer3pos.selected.words')
    ms2lda.plot_lda_fragments(consistency=0.50)

    # test on beer2pos
    
    old_model = CollapseGibbsLda.load('input/beer3pos.model.p')
    if hasattr(old_model, 'selected_topics'):
        print "Persistent topics = " + str(old_model.selected_topics)
 
    relative_intensity = True
    fragment_filename = 'input/relative_intensities/Beer_2_T10_POS_fragments_rel.csv'
    neutral_loss_filename = 'input/relative_intensities/Beer_2_T10_POS_losses_rel.csv'
    mzdiff_filename = None    
    ms1_filename = 'input/relative_intensities/Beer_2_T10_POS_ms1_rel.csv'
    ms2_filename = 'input/relative_intensities/Beer_2_T10_POS_ms2_rel.csv'
 
    ms2lda = Ms2Lda(fragment_filename, neutral_loss_filename, mzdiff_filename, 
                ms1_filename, ms2_filename, relative_intensity)    
    df, vocab = ms2lda.preprocess()    
    ms2lda.run_lda(df, vocab, n_topics, n_samples, n_burn, n_thin, 
                   alpha, beta, use_own_model=True, use_native=True, previous_model=old_model)
    ms2lda.write_results('beer2pos')
    ms2lda.plot_lda_fragments(consistency=0.50)

if __name__ == "__main__": main()