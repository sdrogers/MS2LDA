import operator
import os
import sys

from pandas.core.frame import DataFrame

import matplotlib.patches as mpatches
import numpy as np
import pylab as plt
import math


class Ms2Lda_Viz(object):
    
    def __init__(self, model, ms1, ms2, docdf, topicdf):
        self.model = model
        self.ms1 = ms1
        self.ms2 = ms2
        self.docdf = docdf
        self.topicdf = topicdf        
    
    def rank_topics(self, sort_by="h_index", selected_topics=None, top_N=None, interactive=False):

        print "Ranking topics ..."
        self.sort_by = sort_by
        if sort_by == 'h_index':
            topic_sort_criteria = self._h_index() 
        elif sort_by == 'in_degree':
            topic_sort_criteria = self._in_degree() 
        print "DONE!"
        
        sorted_topic_counts = sorted(topic_sort_criteria.items(), key=operator.itemgetter(1), reverse=True)
        if not interactive:
            print "Topic Ranking"
            print "============="
            print
        counter = 0
        for topic_number, score in sorted_topic_counts:
            if selected_topics is not None:
                if topic_number not in selected_topics:
                    continue
            if not interactive:
                if sort_by == 'h_index':
                    print "Topic " + str(topic_number) + " h-index=" + str(score)
                elif sort_by == 'in_degree':
                    print "Topic " + str(topic_number) + " in-degree=" + str(score)
            counter += 1
            if top_N is not None and counter >= top_N:
                break
        if not interactive:
            print

        # also create a matrix of topics and their h-indices for later use in front-end visualisation
        if interactive:        
            num_topics = len(sorted_topic_counts)
            topic_degree = self._in_degree()
            self.topic_ranking = np.zeros((num_topics, 3), dtype=int)
            for k in range(num_topics):
                item = sorted_topic_counts[k]
                topic_id = item[0]
                self.topic_ranking[k, 0] = topic_id
                self.topic_ranking[k, 1] = item[1] # the h-index or in-degree of that topic
                self.topic_ranking[k, 2] = topic_degree[topic_id]
            self.sort_by_min = np.min(self.topic_ranking[:, 2])
            self.sort_by_max = np.max(self.topic_ranking[:, 2])
        
        return topic_sort_criteria, sorted_topic_counts                
                
    def plot_lda_fragments(self, consistency=0.50, sort_by="h_index", 
                           selected_topics=None, interactive=False,
                           to_highlight=None):
        
        self.to_highlight = to_highlight
                
        if selected_topics is not None and interactive:
            raise ValueError("For interactive mode, the selected_topics parameter is not yet supported so you must visualise all topics.")
                
        topic_ranking, sorted_topic_counts = self.rank_topics(sort_by=sort_by, 
                                                              selected_topics=selected_topics, interactive=interactive)               
        self.topic_plots = {}
        self.topic_ms1_count = {}
        self.topic_ms1_ordering = {}
        self.topic_coordinates = {}
        for (i, c) in sorted_topic_counts:
            
            # skip non-selected topics
            if selected_topics is not None:
                if i not in selected_topics:
                    continue

            if not interactive:            
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
            
            if not interactive:            
                if len(top_n_docs) == 0:
                    print "No parent peaks above the threshold found for this topic"
                    continue
                print "Parent peaks"
                print
                print '     %s\t%s\t\t%s\t\t%s\t\t%s' % ('peakID', 'mz', 'rt', 'int', 'score')
#             else:
#                 if len(top_n_docs) == 0:
#                     continue

            parent_ids = []
            parent_masses = []
            parent_rts = []
            parent_intensities = []
            parent_all_fragments = {}
            parent_word_counts = []
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
    
                if not interactive:
                    print '%-5d%-5d\t%3.5f\t%6.3f\t\t%.3e\t%.3f' % (count, peakid, mz, rt, intensity, prob)
                parent_ids.append(peakid)
                parent_masses.append(mz)
                parent_rts.append(rt)
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
            
            if len(parent_masses) > 0:
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
                    
            if not interactive:
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

                if not interactive:
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

            if not interactive:
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

                if not interactive:
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
    
            if not interactive:
                print
                sys.stdout.flush()
    
            # plot the n_docs parent peaks in this topic            
            # make plot for every parent peak
            num_peaks = len(parent_ids)
            if not interactive:
            
                for n in range(num_peaks):
                        self._make_ms1_plot(i, n, parent_masses, parent_rts, parent_ids, 
                           parent_all_fragments, parent_topic_fragments, parent_topic_losses,
                           wordfreq, consistency, max_parent_mz)
                        plt.show()

            else: # interactive mode

                # set MS1 plot data
                plot_data = (parent_masses, parent_rts, parent_ids, 
                           parent_all_fragments, parent_topic_fragments, parent_topic_losses,
                           wordfreq, consistency, max_parent_mz)
                self.topic_plots[i] = plot_data
                self.topic_ms1_count[i] = num_peaks
                
                print "Generating plots for topic " + str(i) + " h-index = " + str(topic_ranking[i]) + " degree = " + str(num_peaks)
                
                # set coordinate of each circle
                x_coord = topic_ranking[i]
                if num_peaks > 0:
                    y_coord = math.log10(num_peaks)   
                else :
                    y_coord = 0
                self.topic_coordinates.update({i:(x_coord, y_coord)})
                
                # count how many words are being plotted (above the consistency ratio) for each parent id
                parent_word_counts = []
                for parent_id in parent_ids:
                    count = 0
                    if parent_id in parent_topic_fragments:        
                        fragments_list = parent_topic_fragments[parent_id]
                        num_peaks = len(fragments_list)
                        for j in range(num_peaks):
                            item = fragments_list[j]
                            word = item[4]                           
                            freq = wordfreq[word]
                            ratio = float(freq)/len(parent_ids)
                            if ratio >= consistency:
                                count += 1
                    if parent_id in parent_topic_losses:        
                        losses_list = parent_topic_losses[parent_id]
                        num_peaks = len(losses_list)
                        for j in range(num_peaks):
                            word = item[4]
                            freq = wordfreq[word]
                            ratio = float(freq)/len(parent_ids)
                            if ratio >= consistency:
                                count += 1
                    parent_word_counts.append(count)
                 
                parent_word_counts = np.array(parent_word_counts)
                ms1_ordering = np.argsort(parent_word_counts)[::-1]
                self.topic_ms1_ordering[i] = ms1_ordering                
                                    
            # break

        # convert topic_coordinates from dictionary to a list of coordinates, sorted by the topic id
        sorted_coords = sorted(self.topic_coordinates.iteritems(), key=lambda key_value: key_value[0])
        self.topic_coordinates = [item[1] for item in sorted_coords]        
        
    def plot_for_web(self, i, n):
        if i in self.topic_plots:
            parent_masses, parent_rts, parent_ids, parent_all_fragments, parent_topic_fragments, parent_topic_losses, wordfreq, consistency, max_parent_mz = self.topic_plots[i]
            fig = self._make_ms1_plot(i, n, parent_masses, parent_rts, parent_ids, 
                                      parent_all_fragments, parent_topic_fragments, parent_topic_losses,
                                      wordfreq, consistency, max_parent_mz)
            return fig
        else:
            return None
        
    def _make_ms1_plot(self, i, n, parent_masses, parent_rts, parent_ids, 
                       parent_all_fragments, parent_topic_fragments, parent_topic_losses,
                       wordfreq, consistency, max_parent_mz):
        
        # get the image content but in the right order
        pos = self.topic_ms1_ordering[i][n]
                
        parent_fontspec = {
            'size':'12', 
            'color':'blue', 
            'weight':'bold'
        }
        fragment_fontspec = {
            'size':'12', 
            'color':'#800000', 
            'weight':'bold'
        }
        loss_fontspec = {
            'size':'12', 
            'color':'green', 
            'weight':'bold'
        }        
        
        figsize=(10, 6)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        #set the bbox for the text. Increase txt_width for wider text.
        txt_width = 20*(plt.xlim()[1] - plt.xlim()[0])
        txt_height = 0.2*(plt.ylim()[1] - plt.ylim()[0])

        ## handle empty topic
        if len(parent_ids) == 0:

            parent_mass = 0
            parent_rt = 0
            
        else:

            # plot the parent peak first
            parent_mass = parent_masses[pos]
            parent_rt = parent_rts[pos]
    
            # TEMPORARILY HARDCODED FOR RELATIVE INTENSITY 
            parent_intensity = 0.25
            plt.plot((parent_mass, parent_mass), (0, parent_intensity), linewidth=2.0, color='b')
            x = parent_mass
            y = parent_intensity
            parent_id = parent_ids[pos]
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
                intensity = item[3]
                plt.plot((mass, mass), (0, intensity), linewidth=1.0, color='#FF9933')
    
            x_data = []
            y_data = []
            line_type = []            
            xlim_upper = max_parent_mz
    
            # plot the fragment peaks in this topic that also occur in this parent peak
            if parent_id in parent_topic_fragments:        
                fragments_list = parent_topic_fragments[parent_id]
                num_peaks = len(fragments_list)
                for j in range(num_peaks):
                    item = fragments_list[j]
                    peakid = item[0]
                    parentid = item[1]
                    mass = item[2]
                    if mass > xlim_upper:
                        xlim_upper = mass
                    intensity = item[3]
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
                    if mass > xlim_upper:
                        xlim_upper = mass
                    intensity = item[3]
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
    
        plt.xlim([0, xlim_upper+100])
        plt.ylim([0, 1.5])

        plt.xlabel('m/z')
        plt.ylabel('relative intensity')                    
        ms1_peak_counts = str(len(parent_masses))
        mz_value = ("%.5f" % parent_mass)
        rt_value = ("%.3f" % parent_rt)
        title = 'Topic ' + str(i) + ' peak ' + str(n+1) + '/' + ms1_peak_counts
        title += ' (m/z=' + mz_value + ' RT=' + rt_value + ")"
        plt.title(title)
        
        blue_patch = mpatches.Patch(color='blue', label='Parent peak')
        yellow_patch = mpatches.Patch(color='#FF9933', label='Fragment peaks')
        red_patch = mpatches.Patch(color='#800000', label='Topic fragment')
        green_patch = mpatches.Patch(color='green', label='Topic loss')                
        plt.legend(handles=[blue_patch, yellow_patch, red_patch, green_patch])                

        return fig
                    
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
    
    # compute the h-index of topics TODO: this only works for fragment and loss words!
    def _h_index(self):
                
        K = self.model.K
        topic_counts = {}

        for i in range(K):
                        
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
                                        
                    # this code is too slow!
                    # count the citation numbers
#                     for cited in fragment_bin_ids:
#                         if cited == 'nan':
#                             continue
#                         else:
#                             if cited in fragment_words:
#                                 fragment_words[cited] = fragment_words[cited] + 1
#                     for cited in loss_bin_ids:
#                         if cited == 'nan':
#                             continue
#                         else:
#                             if cited in loss_words:
#                                 loss_words[cited] = loss_words[cited] + 1

                    # convert to dictionary for quick lookup
                    word_dict = {}
                    for word in fragment_bin_ids:
                        word_dict.update({word:word})
                    for word in loss_bin_ids:
                        word_dict.update({word:word})

                    # count the citation numbers                                
                    for word in fragment_words:
                        if word in word_dict:
                            fragment_words[word] = fragment_words[word] + 1
                    for word in loss_words:
                        if word in word_dict:
                            loss_words[word] = loss_words[word] + 1
                    
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

                print " - topic " + str(i) + " h-index = " + str(h_index)
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