import cPickle
import gzip
import os
import re
import sys
import time
import timeit
import math

import numpy as np
import pandas as pd
import pylab as plt
import yaml
from scipy.sparse import coo_matrix
import scipy.cluster.hierarchy as hierarchy

from lda_cgs import CollapseGibbsLda
from visualisation.pylab.lda_for_fragments_viz import Ms2Lda_Viz
import visualisation.pyLDAvis as pyLDAvis
import visualisation.sirius.sirius_wrapper as sir
import lda_utils as utils
from efcompute.ef_assigner import ef_assigner
from visualisation.networkx import lda_visualisation

class Ms2Lda(object):
    
    def __init__(self, df, vocab, ms1, ms2, input_filenames=[], EPSILON=0.05):
        self.df = df
        self.vocab = vocab
        self.ms1 = ms1
        self.ms2 = ms2
        self.EPSILON = EPSILON
        self.input_filenames = input_filenames
        
    @classmethod
    def run_feature_extraction(cls, script_folder, config_filename):

        from rpy2.robjects import r
        
        print "script_folder = " + script_folder
        print "configuration filename = " + config_filename
       
        try:
                    
            # call the workflow for feature extraction using rpy2
            print "Running feature extraction in R"
            commands = []
            commands.append("setwd('" + script_folder + "')")
            commands.append("source('startFeatureExtraction.R')")
            commands.append("start_feature_extraction('" + config_filename + "')")
            r['options'](warn=-1)
            for c in commands:
                r(c)

            # load yaml config file
            print "Loading input files"
            with open(config_filename, 'r') as input_file:
     
                # get file prefix
                config = yaml.load(input_file)
                prefix = config['input_files']['prefix']
     
                # construct path to each input file
                fragment_filename = os.path.join(script_folder, prefix + '_fragments.csv')
                neutral_loss_filename = os.path.join(script_folder, prefix + '_losses.csv')
                mzdiff_filename = None
                ms1_filename = os.path.join(script_folder, prefix + '_ms1.csv')
                ms2_filename = os.path.join(script_folder, prefix + '_ms2.csv')

                print
                print "Feature extraction done"
                print "fragment_filename = " + fragment_filename
                print "neutral_loss_filename = " + neutral_loss_filename
                print "mzdiff_filename = " + str(mzdiff_filename)
                print "ms1_filename = " + ms1_filename
                print "ms2_filename = " + ms2_filename
                return Ms2Lda.lcms_data_from_R(fragment_filename, neutral_loss_filename, mzdiff_filename,
                                 ms1_filename, ms2_filename)

        except Exception as e:
            print "Exception caught: " + str(e)

    @classmethod
    def lcms_data_from_R(cls, fragment_filename, neutral_loss_filename, mzdiff_filename, 
                 ms1_filename, ms2_filename, vocab_type=1):

        print "Loading input files"
        input_filenames = []
        fragment_data = None
        neutral_loss_data = None
        mzdiff_data = None

        # load all the input files        
        if fragment_filename is not None:
            fragment_data = pd.read_csv(fragment_filename, index_col=0)
            input_filenames.append(fragment_filename)
        if neutral_loss_filename is not None:
            neutral_loss_data = pd.read_csv(neutral_loss_filename, index_col=0)
            input_filenames.append(neutral_loss_filename)
        if mzdiff_filename is not None:
            mzdiff_data = pd.read_csv(mzdiff_filename, index_col=0)
            input_filenames.append(mzdiff_filename)
        
        ms1 = pd.read_csv(ms1_filename, index_col=0)
        ms2 = pd.read_csv(ms2_filename, index_col=0)
        input_filenames.append(ms1_filename)
        input_filenames.append(ms2_filename)
        
        ms2['fragment_bin_id'] = ms2['fragment_bin_id'].astype(str)
        ms2['loss_bin_id'] = ms2['loss_bin_id'].astype(str)

        data = pd.DataFrame()

        # discretise the fragment and neutral loss intensities values by converting it to 0 .. 100
        if fragment_data is not None:
            fragment_data *= 100
            data = data.append(fragment_data)
        
        if neutral_loss_data is not None:
            neutral_loss_data *= 100
            data = data.append(neutral_loss_data)
        
        # make mzdiff values to be within 0 .. 100 as well
        if mzdiff_data is not None:
            max_mzdiff_count = mzdiff_data.max().max()
            mzdiff_data /= max_mzdiff_count
            mzdiff_data *= 100
            data = data.append(mzdiff_data)            
        
        # get rid of NaNs, transpose the data and floor it
        data = data.replace(np.nan,0)
        data = data.transpose()
        sd = coo_matrix(data)
        sd = sd.floor()  
        npdata = np.array(sd.todense(), dtype='int32')
        print "Data shape " + str(npdata.shape)
        df = pd.DataFrame(npdata)
        df.columns = data.columns
        df.index = data.index

        # decide how to generate vocab
        if vocab_type == 1:
            # vocab is just a string of the column names
            vocab = data.columns.values 
        elif vocab_type == 2:
            # vocab is a tuple of (column name, word_type)
            all_words = data.columns.values
            vocab = []
            for word in all_words:
                if word.startswith('fragment'):
                    word_type = 0
                elif word.startswith('loss'):
                    word_type = 1
                elif word.startswith('mzdiff'):
                    word_type = 2
                else:
                    raise ValueError("Unknown word type")
                tup = (word, word_type)
                vocab.append(tup)
            vocab = np.array(vocab)
        else:
            raise ValueError("Unknown vocab type")

        # return the instantiated object        
        this_instance = cls(df, vocab, ms1, ms2, input_filenames)        
        return this_instance         
    
    @classmethod
    def resume_from(cls, project_in, verbose=True):
        start = timeit.default_timer()        
        with gzip.GzipFile(project_in, 'rb') as f:
            obj = cPickle.load(f)
            stop = timeit.default_timer()
            if verbose:
                print "Project loaded from " + project_in + " time taken = " + str(stop-start)
                print " - input_filenames = "
                for fname in obj.input_filenames:
                    print "\t" + fname
                print " - df.shape = " + str(obj.df.shape)
                if hasattr(obj, 'model'):
                    print " - K = " + str(obj.model.K)
                    print " - alpha = " + str(obj.model.alpha[0])
                    print " - beta = " + str(obj.model.beta[0])
                    print " - number of samples stored = " + str(len(obj.model.samples))
                else:
                    print " - No LDA model found"
                print " - last_saved_timestamp = " + str(obj.last_saved_timestamp)  
                if hasattr(obj, 'message'):
                    print " - message = " + str(obj.message)  
            return obj  
        
    @classmethod
    def gcms_data_from_mzmatch(cls, input_filename, intensity_colname, tol):

        # load the data, using the column indicated by intensity_colname as the intensity values
        df = pd.DataFrame.from_csv(input_filename, sep='\t');
        mass = df.index.tolist()
        rt = df['RT'].tolist() # assume the input file always has this column
        intensity = df[intensity_colname].tolist()
        rid = df['relation.id'].tolist() # assume the input file always has this column
                        
        # Group fragments if they are within tol ppm of each other
        unique_masses = []
        mass_id = []
        for m in mass:
            # check for previous
            previous_pos = [i for i,a in enumerate(unique_masses) if (abs(m-a)/m)*1e6 < tol]
            if len(previous_pos) == 0:
                # it's a new one
                unique_masses.append(m)
                mass_id.append(len(unique_masses)-1)
            else:
                # it's an old one
                mass_id.append(previous_pos[0])        

        # create some dummy MS1 peaklist
        ms1_peakids = list(set(rid))
        ms1_peakdata = []
        for pid in ms1_peakids:
            ms1_peakdata.append({'peakID': pid, 'MSnParentPeakID': 0, 'msLevel': 1, \
                                 'rt': 0.0, 'mz': 300.0, 'intensity': 3.0E5})
        ms1 = pd.DataFrame(ms1_peakdata, index=ms1_peakids)
                
        # create the MS2 peaklist
        n_peaks = len(mass)
        pid = max(ms1_peakids)+1
        ms2_peakids = []
        ms2_peakdata = []
        for n in range(n_peaks):
            ms2_peakdata.append({'peakID': pid, 'MSnParentPeakID': rid[n], 'msLevel': 2, \
                                 'rt': rt[n], 'mz': mass[n], 'intensity': intensity[n], \
                                 'fragment_bin_id': str(unique_masses[mass_id[n]]), \
                                 'loss_bin_id': np.nan})
            ms2_peakids.append(pid)
            pid += 1
        ms2 = pd.DataFrame(ms2_peakdata, index=ms2_peakids)

        # Create the data matrix and then trim to get rid of rare fragments, and dodgy data items
        dmat = np.zeros((len(unique_masses),max(rid)+1),np.float)
        for i,m in enumerate(mass):
            dmat[mass_id[i],rid[i]] = intensity[i]
            
        min_met = 2
        r,c = dmat.shape
        remove = []
        col_names = np.array(range(max(rid)+1))
        row_names = np.array(unique_masses)
        for i in range(r):
            s = np.where(dmat[i,:]>0)[0]
            if len(s)<min_met:
                remove.append(i)
        
        remove = np.array(remove)
        row_names = np.delete(row_names,remove)
        dmat = np.delete(dmat,remove,axis=0)
        min_frag = 3
        r,c = dmat.shape
        remove = []
        for i in range(c):
            s = np.where(dmat[:,i]>0)[0]
            if len(s)<min_frag:
                remove.append(i)
        remove = np.array(remove)
        
        col_names = np.delete(col_names,remove)
        dmat = np.delete(dmat,remove,axis=1)
        
        # Remove fragments that appear nowhere
        remove = []
        for i in range(r):
            s = np.where(dmat[i,:]>0)[0]
            if len(s) == 0:
                remove.append(i)
        
        dmat = np.delete(dmat,remove,axis=0)
        row_names = np.delete(row_names,remove)
        
        print dmat.shape,row_names.shape,col_names.shape   
        
        # Turn into integer array with biggest peak in each spectra at 100
        dmat_int = np.zeros(dmat.shape,np.int)
        r,c = dmat.shape
        for i in range(c):
            ma = dmat[:,i].max()
            dmat_int[:,i] = 100*dmat[:,i]/ma
            
        # Make into Pandas structure
        row_names = ['fragment_' + str(x) for x in row_names]
        col_names = ['300_0_' + str(x) for x in col_names]
        df = pd.DataFrame(dmat_int,index=row_names,columns = col_names)
        df = df.transpose()
        vocab = df.columns
                
        # return the instantiated object        
        input_filenames = [input_filename]
        this_instance = cls(df, vocab, ms1, ms2, input_filenames)
        return this_instance        
        
    def run_lda(self, n_topics, n_samples, n_burn, n_thin, alpha, beta, 
                use_native=True, previous_model=None):    
                        
        print "Fitting model..."
        self.n_topics = n_topics
        self.model = CollapseGibbsLda(self.df, self.vocab, n_topics, alpha, beta, previous_model=previous_model)
        self.n_topics = self.model.K # might change if previous_model is used

        start = timeit.default_timer()
        self.model.run(n_burn, n_samples, n_thin, use_native=use_native)
        stop = timeit.default_timer()
        print "DONE. Time=" + str(stop-start)
        
    def do_thresholding(self, th_doc_topic=0.05, th_topic_word=0.0):

        # save the thresholding values used for visualisation later
        self.th_doc_topic = th_doc_topic
        self.th_topic_word = th_topic_word
        
        previous_model = self.model.previous_model
        selected_topics = None
        if previous_model is not None and hasattr(previous_model, 'selected_topics'):
            selected_topics = previous_model.selected_topics
            
        # get rid of small values in the matrices of the results
        # if epsilon > 0, then the specified value will be used for thresholding
        # otherwise, the smallest value for each row in the matrix is used instead
        self.topic_word = utils.threshold_matrix(self.model.topic_word_, epsilon=th_topic_word)
        self.doc_topic = utils.threshold_matrix(self.model.doc_topic_, epsilon=th_doc_topic)
        
        self.topic_names = []
        counter = 0
        for i, topic_dist in enumerate(self.topic_word):
            if selected_topics is not None:
                if i < len(selected_topics):
                    topic_name = 'Fixed_M2M {}'.format(selected_topics[i])
                else:
                    topic_name = 'M2M_{}'.format(counter)
                    counter += 1
            else:
                topic_name = 'M2M_{}'.format(i)                    
            self.topic_names.append(topic_name)
        
        # create document-topic output file        
        masses = np.array(self.df.transpose().index)
        d = {}
        for i in np.arange(self.n_topics):
            topic_name = self.topic_names[i]
            topic_series = pd.Series(self.topic_word[i], index=masses)
            d[topic_name] = topic_series
        self.topicdf = pd.DataFrame(d)
        
        # make sure that columns in topicdf are in the correct order
        # because many times we'd index the columns in the dataframes directly by their positions
        cols = self.topicdf.columns.tolist()
        sorted_cols = self._natural_sort(cols)
        self.topicdf = self.topicdf[sorted_cols]        
    
        # create topic-docs output file
        (n_doc, a) = self.doc_topic.shape
        topic_index = np.arange(self.n_topics)
        doc_names = np.array(self.df.index)
        d = {}
        for i in np.arange(n_doc):
            doc_name = doc_names[i]
            doc_series = pd.Series(self.doc_topic[i], index=topic_index)
            d[doc_name] = doc_series
        self.docdf = pd.DataFrame(d)
        
        # sort columns by mass_rt values
        cols = self.docdf.columns.tolist()
        mass_rt = [(float(m.split('_')[0]),float(m.split('_')[1])) for m in cols]
        sorted_mass_rt = sorted(mass_rt,key=lambda m:m[0])
        ind = [mass_rt.index(i) for i in sorted_mass_rt]
        self.docdf = self.docdf[ind]
        # self.docdf.to_csv(outfile)se

#         # threshold docdf to get rid of small values and also scale it
#         for i, row in self.docdf.iterrows(): # iterate through the rows
#             doc = self.docdf.ix[:, i]
#             selected = doc[doc>0]
#             count = len(selected.values)
#             selected = selected * count
#             self.docdf.ix[:, i] = selected
        self.docdf = self.docdf.replace(np.nan, 0)
                                
    def write_results(self, results_prefix):
        
        if not hasattr(self, 'topic_word'):
            raise ValueError('Thresholding not done yet.')
        
        # create topic-word output file
        outfile = self._get_outfile(results_prefix, '_motifs.csv') 
        print "Writing Mass2Motif features to " + outfile
        with open(outfile,'w') as f:
            
            for i, topic_dist in enumerate(self.topic_word):

                ordering = np.argsort(topic_dist)
                vocab = self.df.columns.values                
                topic_words = np.array(vocab)[ordering][::-1]
                dist = topic_dist[ordering][::-1]
                topic_name = self.topic_names[i]
                f.write(topic_name)
                
                # filter entries to display
                for j in range(len(topic_words)):
                    if dist[j] > 0:
                        f.write(',{}'.format(topic_words[j]))
                    else:
                        break
                f.write('\n')
    
        # write out topicdf and docdf

        outfile = self._get_outfile(results_prefix, '_features.csv') 
        print "Writing features X motifs to " + outfile
        self.topicdf.to_csv(outfile)
    
        outfile = self._get_outfile(results_prefix, '_docs.csv') 
        print "Writing docs X motifs to " + outfile
        docdf = self.docdf.transpose()
        docdf.columns = self.topic_names
        docdf.to_csv(outfile)
        
    def save_project(self, project_out, message=None):
        start = timeit.default_timer()        
        self.last_saved_timestamp = str(time.strftime("%c"))
        self.message = message
        with gzip.GzipFile(project_out, 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
            stop = timeit.default_timer()
            print "Project saved to " + project_out + " time taken = " + str(stop-start)
        
    def persist_topics(self, topic_indices, model_out, words_out):
        self.model.save(topic_indices, model_out, words_out)
        
    def rank_topics(self, sort_by="h_index", selected_topics=None, top_N=None):
        plotter = Ms2Lda_Viz(self.model, self.ms1, self.ms2, self.docdf, self.topicdf)
        return plotter.rank_topics(sort_by=sort_by, selected_topics=selected_topics, top_N=top_N)
        
    def plot_lda_fragments(self, selected_motifs=None, interactive=False, to_highlight=None, 
                           additional_info={}):

        # these used to be user-defined parameters, but now they're fixed
        consistency=0.0 # TODO: remove this
        sort_by="h_index"

        if not hasattr(self, 'topic_word'):
            raise ValueError('Thresholding not done yet.')        
        
        plotter = Ms2Lda_Viz(self.model, self.ms1, self.ms2, self.docdf, self.topicdf)
        if interactive:
            # if interactive mode, we always sort by the h_index because we need both the h-index and degree for plotting
            plotter.plot_lda_fragments(consistency=consistency, sort_by='h_index', 
                                       selected_motifs=selected_motifs, interactive=interactive,
                                       to_highlight=to_highlight)
            # self.model.visualise(plotter)
            data = {}
            data['topic_term_dists'] = self.model.topic_word_
            data['doc_topic_dists'] = self.model.doc_topic_
            data['doc_lengths'] = self.model.cd
            data['vocab'] = self.model.vocab
            data['term_frequency'] = np.sum(self.model.ckn, axis=0)    
            data['topic_ranking'] = plotter.topic_ranking
            data['topic_coordinates'] = plotter.topic_coordinates
            data['plot_opts'] = {'xlab': 'h-index', 'ylab': 'log(degree)', 'sort_by' : plotter.sort_by}
            data['lambda_step'] = 5         
            data['lambda_min'] = utils.round_nicely(plotter.sort_by_min)
            data['lambda_max'] = utils.round_nicely(plotter.sort_by_max)
            data['th_topic_word'] = self.th_topic_word
            data['th_doc_topic'] = self.th_doc_topic
            data['topic_wordfreq'] = plotter.topic_wordfreqs
            data['topic_ms1_count'] = plotter.topic_ms1_count
            data['topic_annotation'] = additional_info
            vis_data = pyLDAvis.prepare(**data)   
            pyLDAvis.show(vis_data, topic_plotter=plotter)
        else:
            plotter.plot_lda_fragments(consistency=consistency, sort_by=sort_by, 
                                       selected_motifs=selected_motifs, interactive=interactive)
            
    def get_network_graph(self, to_highlight=None, degree_filter=0, selected_motifs=None):
        plotter = Ms2Lda_Viz(self.model, self.ms1, self.ms2, self.docdf, self.topicdf)
        json_data, G = lda_visualisation.get_json_from_docdf(plotter.docdf.transpose(), to_highlight, degree_filter, selected_motifs=selected_motifs)
        return G, json_data

    # this should only be run once LDA has been run and the thresholding applied,
    # because docdf wouldn't exist otherwise            
    def run_cosine_clustering(self, method='greedy', th_clustering=0.55):
        
        if not hasattr(self, 'topic_word'):
            raise ValueError('Thresholding not done yet.')        
        
        # Swap the NaNs for zeros. Turn into a numpy array and grab the parent names
        data = self.docdf.fillna(0)
        data_array = np.array(data)
        peak_names = list(data.columns.values)

        # Create a matrix with the normalised values (each parent ion has magnitude 1)
        l = np.sqrt((data_array**2).sum(axis=0))
        norm_data = np.divide(data_array,l)

        if method.lower() == 'hierarchical': # scipy hierarchical clustering
        
            clustering = hierarchy.fclusterdata(norm_data.transpose(), th_clustering, criterion = 'distance', 
                                                metric='euclidean', method='single')
        
        elif method.lower() == 'greedy': # greedy cosine clustering
        
            cosine_sim = np.dot(norm_data.transpose(),norm_data)
            finished = False
            total_intensity = data_array.sum(axis=0)
            total_intensity = total_intensity
            n_features, n_parents = data_array.shape
            clustering = np.zeros((n_parents,),np.int)
            current_cluster = 1
            thresh = th_clustering
            count = 0
            while not finished:
                # Find the parent with the max intensity left
                current = np.argmax(total_intensity)
                total_intensity[current] = 0.0
                count += 1
                clustering[current] = current_cluster
                # Find other parents with cosine similarity over the threshold
                friends = np.where((cosine_sim[current,:]>thresh) * (total_intensity > 0.0))[0]
                clustering[friends] = current_cluster
                total_intensity[friends] = 0.0
                # When points are clustered, their total_intensity is set zto zero. 
                # If there is nothing left with zero, quit
                left = np.where(total_intensity > 0.0)[0]
                if len(left) == 0:
                    finished = True
                current_cluster += 1
                    
        else:
            raise ValueError('Unknown clustering method')
        
        return peak_names, clustering
    
    def plot_cosine_clustering(self, motif_id, clustering, peak_names):  
        
        if not hasattr(self, 'topic_word'):
            raise ValueError('Thresholding not done yet.')        
        
        colnames = self.docdf.columns.values
        row = self.docdf.iloc[[motif_id]]
        pos = row.values[0] > 0
        ions_of_interest = colnames[pos]
        
        plotter = Ms2Lda_Viz(self.model, self.ms1, self.ms2, self.docdf, self.topicdf)
        G, cluster_interests = plotter.plot_cosine_clustering(motif_id, ions_of_interest, clustering, peak_names)
        return G, cluster_interests

    def print_topic_words(self, selected_topics=None, with_probabilities=True, compact_output=False):
        
        raise ValueError("print_topic_words is now called print_motif_features")
            
    def print_motif_features(self, selected_motifs=None, with_probabilities=True, quiet=False):
        
        if not hasattr(self, 'topic_word'):
            raise ValueError('Thresholding not done yet.')
        
        word_map = {}
        topic_map = {}
        for i, topic_dist in enumerate(self.topic_word):    

            show_print = False
            if selected_motifs is None:
                show_print = True
            if selected_motifs is not None and i in selected_motifs:
                show_print = True
                
            if show_print:
                ordering = np.argsort(topic_dist)
                topic_words = np.array(self.vocab)[ordering][::-1]
                dist = topic_dist[ordering][::-1]        
                topic_name = 'Mass2Motif {}:'.format(i)
                front = topic_name
                back = ""                    
                for j in range(len(topic_words)):
                    if dist[j] > 0:
                        single_word = topic_words[j]
                        if single_word in word_map:
                            word_map[single_word].add(i)
                        else:
                            word_map[single_word] = set([i])
                        if with_probabilities:
                            back += '%s (%.3f),' % (single_word, dist[j])
                        else:
                            back += '%s,' % (single_word)
                    else:
                        break
                topic_map[i] = back
                if not quiet:
                    output = front + back
                    print output
        return word_map, topic_map
        
    def plot_posterior_alpha(self):
        posterior_alpha = self.model.posterior_alpha
        posterior_alpha = posterior_alpha / np.sum(posterior_alpha)
        ind = range(len(posterior_alpha))
        plt.bar(ind, posterior_alpha, 2)
        
    def annotate_with_sirius(self, sirius_platform="orbitrap", mode="pos", ppm_max=5, min_score=0.01, max_ms1=700, 
                             verbose=False):
        mode = mode.lower()
        annot_ms1, annot_ms2 = sir.annotate_sirius(self.ms1, self.ms2, sirius_platform=sirius_platform, 
                                                   mode=mode, ppm_max=ppm_max, min_score=min_score, 
                                                   max_ms1=max_ms1, verbose=verbose)
        self.ms1 = annot_ms1
        self.ms2 = annot_ms2

    def annotate_peaks(self, mode="pos", target="ms2_fragment", ppm=5, 
                       scale_factor=1000, max_mass=200, n_stages=1,
                       rule_8_max_occurrences=None, verbose=False):

        self._check_valid_input(mode, target, ppm)        
        self._print_annotate_banner(target, mode, ppm, scale_factor, max_mass)
        
        ## override with sensible values
        if target == 'ms2_loss':
            mode = 'none'

        # will return different mass list, depending on whether it's for MS1 parents, 
        # MS2 fragments or MS2 losses
        mass_list = self._get_mass_list(target)

        # run first-stage EF annotation on the mass list
        ef = ef_assigner(scale_factor=scale_factor, do_7_rules=True, 
                         second_stage=False, rule_8_max_occurrences=rule_8_max_occurrences)
        _, top_hit_string, _ = ef.find_formulas(mass_list, ppm=ppm, polarisation=mode.upper(), 
                                                max_mass_to_check=max_mass)
        assert len(mass_list) == len(top_hit_string)

        # anything that's None is to be annotated again for the second stage
        if n_stages == 2:

            mass_list_2 = []
            to_process_idx = []
            for n in range(len(mass_list)):
                mass = mass_list[n]
                tophit = top_hit_string[n]
                if tophit is None:
                    mass_list_2.append(mass)
                    to_process_idx.append(n)
                
            print
            print "=================================================================="
            print "Found " + str(len(mass_list_2)) + " masses for second-stage EF annotation"
            print "=================================================================="
            print
    
            # run second-stage EF annotation        
            ef = ef_assigner(scale_factor=scale_factor, do_7_rules=True, 
                             second_stage=True, rule_8_max_occurrences=rule_8_max_occurrences)
            _, top_hit_string_2, _ = ef.find_formulas(mass_list_2, ppm=ppm, polarisation=mode.upper(), 
                                                      max_mass_to_check=max_mass)
            
            # copy 2nd stage result back to the 1st stage result
            for i in range(len(top_hit_string_2)):
                n = to_process_idx[i]
                top_hit_string[n] = top_hit_string_2[i
                                                 ]        
        # set the results back
        self._set_annotation_results(target, mass_list, top_hit_string)        

    def _check_valid_input(self, mode, target, ppm_list):
        ''' Checks EF annotation input parameters are valid ''' 

        ## Checks mode is valid
        mode = mode.lower()
        if mode != "pos" and mode != "neg" and mode != 'none':
            raise ValueError("mode is either 'pos', 'neg' or 'none'")        

        ## Checks target is valid
        target = target.lower()
        if target != "ms1" and target != "ms2_fragment" and target != 'ms2_loss':
            raise ValueError("target is either 'ms1', 'ms2_fragment' or 'ms2_loss'")        
        
        ## Checks if it's a conditional ppm list then it's in a valid format
        if type(ppm_list) is list:                

            # check length
            if len(ppm_list) != 2:
                raise ValueError("The list of conditional ppm values is not valid. Valid example: [(80, 5), (200, 10)]")
            
            # check items are in the right order
            prev = 0
            for item in ppm_list:
                mass = item[0]
                if mass < prev:
                    raise ValueError("The list of conditional ppm values is in the right order. Valid example: [(80, 5), (200, 10)]")
                prev = mass
            
    def _print_annotate_banner(self, title, mode, ppm, scale_factor, max_mass):
        print "***********************************"   
        print "Annotating " + title
        print "***********************************"   
        print
        print "- mode = " + mode
        print "- ppm = " + str(ppm)
        print "- scale_factor = " + str(scale_factor)
        print "- max_mass = " + str(max_mass)
        print        
        sys.stdout.flush()
        
    def _get_mass_list(self, target):
        ''' Retrieves a different mass list, depending on the target
        (whether it's for ms1 or ms2 fragment or ms2 loss annotation)'''

        if target == 'ms1':

            # use the masses from the MS1 peaklist
            mass_list = self.ms1.mz.values.tolist()
        
        elif target == 'ms2_fragment':

            # use the fragment bins, rather than the actual MS2 peaklist
            mass_list = self.ms2.fragment_bin_id.values.tolist()
            for n in range(len(mass_list)):
                mass_list[n] = float(mass_list[n])
            mass_list = sorted(set(mass_list))
        
        elif target == 'ms2_loss':

            # use the loss bins, rather than the actual MS2 loss values
            from_dataframe = self.ms2.loss_bin_id.values.tolist()
            mass_list = []
            for mass in from_dataframe:
                mass = float(mass)
                if not math.isnan(mass):
                    mass_list.append(mass)
            mass_list = sorted(set(mass_list))

        return mass_list        

    def _set_annotation_results(self, target, mass_list, top_hit_string):
        ''' Writes annotation results back into the right dataframe column '''

        if target == 'ms1': # set the results back into the MS1 dataframe

            # replace all formulae from None to NaN
            for i in range(len(top_hit_string)):
                if top_hit_string[i] is None:
                    top_hit_string[i] = np.NaN

            self.ms1['annotation'] = top_hit_string        

        elif target == 'ms2_fragment' or target == 'ms2_loss':

            # annotation doesn't exist, set new annotation column
            new_column = False
            if 'annotation' not in self.ms2.columns:
                self.ms2['annotation'] = np.NaN
                new_column = True
            
            for n in range(len(mass_list)):
            
                # write to the annotation column in the dataframe for all MS2 having this fragment or loss bin
                mass_str = str(mass_list[n])
                if target == 'ms2_fragment':
                    members = self.ms2[self.ms2.fragment_bin_id==mass_str]
                elif target == 'ms2_loss':
                    members = self.ms2[self.ms2.loss_bin_id==mass_str]

                for row_index, row in members.iterrows():
                    formula = top_hit_string[n]
                    if new_column:
                        # annotation column is empty for this row, so overwrite it
                        if formula is None:
                            formula = np.NaN
                        elif target == 'ms2_loss':
                            formula = "loss_" + formula                                
                        self.ms2.loc[row_index, 'annotation'] = formula
                    else:
                        # annotation column already exists
                        if formula is not None:

                            if target == 'ms2_loss':
                                formula = "loss_" + formula
                            
                            current_val = self.ms2.loc[row_index, 'annotation']
                            try: # detect NaN
                                parsed_val = float(current_val)
                                if np.isnan(parsed_val):
                                    append = False # if NaN then overwrite
                            except ValueError:
                                parsed_val = current_val 
                                append = True # otherwise append to the existing annotation value
                            
                            if append:
                                self.ms2.loc[row_index, 'annotation'] += ',' + formula
                            else:
                                self.ms2.loc[row_index, 'annotation'] = formula

    def remove_all_annotations(self):
        ''' Clears all EF annotations from the dataframes '''
        if 'annotation' in self.ms1.columns:        
            self.ms1.drop('annotation', inplace=True, axis=1)        
        if 'annotation' in self.ms2.columns:        
            self.ms2.drop('annotation', inplace=True, axis=1)        
        
    def plot_log_likelihood(self):
        plt.plot(self.model.loglikelihoods_)        
            
    def _natural_sort(self, l): 
        convert = lambda text: int(text) if text.isdigit() else text.lower() 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(l, key = alphanum_key)
        
    def _get_outfile(self, results_prefix, doctype):
        parent_dir = 'results/' + results_prefix
        outfile = parent_dir + '/' + results_prefix + doctype
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)        
        return outfile             
            
def test_lda():

    if len(sys.argv)>1:
        n_topics = int(sys.argv[1])
    else:
        n_topics = 30

    n_topics = 300
    n_samples = 3
    n_burn = 0
    n_thin = 1
    alpha = 50.0/n_topics
    beta = 0.1

    # train on beer3pos
    
#     fragment_filename = 'input/relative_intensities/Beer_3_T10_POS_fragments_rel.csv'
#     neutral_loss_filename = 'input/relative_intensities/Beer_3_T10_POS_losses_rel.csv'
#     mzdiff_filename = None    
#     ms1_filename = 'input/relative_intensities/Beer_3_T10_POS_ms1_rel.csv'
#     ms2_filename = 'input/relative_intensities/Beer_3_T10_POS_ms2_rel.csv'
  
    fragment_filename = 'input/test_mz_rt_pairs_Beer2_withIntensities_fragments.csv'
    neutral_loss_filename = 'input/test_mz_rt_pairs_Beer2_withIntensities_losses.csv'
    mzdiff_filename = None    
    ms1_filename = 'input/test_mz_rt_pairs_Beer2_withIntensities_ms1.csv'
    ms2_filename = 'input/test_mz_rt_pairs_Beer2_withIntensities_ms2.csv'  
  
    ms2lda = Ms2Lda.lcms_data_from_R(fragment_filename, neutral_loss_filename, mzdiff_filename, 
                                     ms1_filename, ms2_filename)    
    ms2lda.run_lda(n_topics, n_samples, n_burn, n_thin, alpha, beta)
#     ms2lda.save_project('results/beer3pos.project')
    
    # new_ms2lda = Ms2Lda.resume_from('results/beer3pos.project')
    
    ms2lda.write_results('beer3pos')
    ms2lda.model.print_topic_words()    
    ms2lda.plot_lda_fragments(consistency=0.50, sort_by="h_index", interactive=True)

# 
#     # save some topics from beer3pos lda
#     
#     selected_topics = [0, 1, 2, 3, 4, 5]    
#     ms2lda.save_model(selected_topics, 'input/beer3pos.model.p', 'input/beer3pos.selected.words')
#     ms2lda.plot_lda_fragments(consistency=0.50)
# 
#     # test on beer2pos
#     
#     old_model = CollapseGibbsLda.load('input/beer3pos.model.p')
#     if hasattr(old_model, 'selected_topics'):
#         print "Persistent topics = " + str(old_model.selected_topics)
#  
#     relative_intensity = True
#     fragment_filename = 'input/relative_intensities/Beer_2_T10_POS_fragments_rel.csv'
#     neutral_loss_filename = 'input/relative_intensities/Beer_2_T10_POS_losses_rel.csv'
#     mzdiff_filename = None    
#     ms1_filename = 'input/relative_intensities/Beer_2_T10_POS_ms1_rel.csv'
#     ms2_filename = 'input/relative_intensities/Beer_2_T10_POS_ms2_rel.csv'
#  
#     ms2lda = Ms2Lda(fragment_filename, neutral_loss_filename, mzdiff_filename, 
#                 ms1_filename, ms2_filename, relative_intensity)    
#     df, vocab = ms2lda.preprocess()    
#     ms2lda.run_lda(df, vocab, n_topics, n_samples, n_burn, n_thin, 
#                    alpha, beta, use_own_model=True, use_native=True, previous_model=old_model)
#     ms2lda.write_results('beer2pos')
#     ms2lda.plot_lda_fragments(consistency=0.50)
                
def main():    
    test_lda()
    
if __name__ == "__main__": main()
