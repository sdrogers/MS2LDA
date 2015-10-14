import cPickle
import gzip
import os
import re
import sys
import time
import timeit

import numpy as np
import pandas as pd
import pylab as plt
from scipy.sparse import coo_matrix
import yaml

from lda_cgs import CollapseGibbsLda
from visualisation.pylab.lda_for_fragments_viz import Ms2Lda_Viz
import visualisation.pyLDAvis as pyLDAvis
import visualisation.sirius.sirius_wrapper as sir
import lda_utils as utils
from efcompute.ef_assigner import ef_assigner

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
    def resume_from(cls, project_in):
        start = timeit.default_timer()        
        with gzip.GzipFile(project_in, 'rb') as f:
            obj = cPickle.load(f)
            stop = timeit.default_timer()
            print "Project loaded from " + project_in + " time taken = " + str(stop-start)
            print " - input_filenames = "
            for fname in obj.input_filenames:
                print "\t" + fname
            print " - df.shape = " + str(obj.df.shape)
            print " - K = " + str(obj.model.K)
            print " - alpha = " + str(obj.model.alpha[0])
            print " - beta = " + str(obj.model.beta[0])
            print " - number of samples stored = " + str(len(obj.model.samples))
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
                    topic_name = 'Fixed Topic {}'.format(selected_topics[i])
                else:
                    topic_name = 'Topic {}'.format(counter)
                    counter += 1
            else:
                topic_name = 'Topic {}'.format(i)                    
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
        outfile = self._get_outfile(results_prefix, '_topics.csv') 
        print "Writing topics to " + outfile
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

        outfile = self._get_outfile(results_prefix, '_all.csv') 
        print "Writing fragments x topics to " + outfile
        self.topicdf.to_csv(outfile)
    
        outfile = self._get_outfile(results_prefix, '_docs.csv') 
        print "Writing topic docs to " + outfile
        self.docdf.transpose().to_csv(outfile)
        
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
        
    def plot_lda_fragments(self, consistency=0.0, sort_by="h_index", 
                           selected_topics=None, interactive=False, to_highlight=None):

        if not hasattr(self, 'topic_word'):
            raise ValueError('Thresholding not done yet.')        
        
        plotter = Ms2Lda_Viz(self.model, self.ms1, self.ms2, self.docdf, self.topicdf)
        if interactive:
            # if interactive mode, we always sort by the h_index because we need both the h-index and degree for plotting
            plotter.plot_lda_fragments(consistency=consistency, sort_by='h_index', 
                                       selected_topics=selected_topics, interactive=interactive,
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
            vis_data = pyLDAvis.prepare(**data)   
            pyLDAvis.show(vis_data, topic_plotter=plotter)
        else:
            plotter.plot_lda_fragments(consistency=consistency, sort_by=sort_by, 
                                       selected_topics=selected_topics, interactive=interactive)
            
    def print_topic_words(self, selected_topics=None, with_probabilities=True, compact_output=False):
        
        if not hasattr(self, 'topic_word'):
            raise ValueError('Thresholding not done yet.')
        
        for i, topic_dist in enumerate(self.topic_word):    

            show_print = False
            if selected_topics is None:
                show_print = True
            if selected_topics is not None and i in selected_topics:
                show_print = True
                
            if show_print:
                ordering = np.argsort(topic_dist)
                topic_words = np.array(self.vocab)[ordering][::-1]
                dist = topic_dist[ordering][::-1]        
                topic_name = 'Topic {}:'.format(i)
                print topic_name,                    
                for j in range(len(topic_words)):
                    if dist[j] > 0:
                        if with_probabilities:
                            print('{} ({}),'.format(topic_words[j], dist[j])),
                        else:
                            print('{},'.format(topic_words[j])),                            
                    else:
                        break
                if compact_output:
                    print
                else:
                    print "\n"
        
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

    def annotate_with_ef_assigner(self, mode="pos", ppm_max=5, scale_factor=1000, max_ms1=700,
                             verbose=False):
        
        mode = mode.lower()
        if mode != "pos" and mode != "neg":
            raise ValueError("mode is either 'pos' or 'neg'")
        else:
            print "Running EF annotation (with 7 golden rules filtering) with parameters:"
            print "- mode = " + mode
            print "- ppm_max = " + str(ppm_max)
            print "- max_ms1 = " + str(max_ms1)
            print        

        # run EF annotation on MS1 dataframe        
        print "Annotating MS1 dataframe"
        mass_list = self.ms1.mz.values.tolist()
        ef = ef_assigner(scale_factor=1000)
        formulas_out, top_hit_string, precursor_mass_list = ef.find_formulas(mass_list, ppm=ppm_max, polarisation=mode.upper(), max_mass_to_check=max_ms1)
        
        # replace all None with NaN
        for i in range(len(top_hit_string)):
            if top_hit_string[i] is None:
                top_hit_string[i] = np.NaN        

        # set the results back into the dataframe        
        self.ms1['annotation'] = top_hit_string
        
        # run EF annotation on MS2 dataframe        
        print "Annotating MS2 dataframe"
        mass_list = self.ms1.mz.values.tolist()
        ef = ef_assigner(scale_factor=1000)
        formulas_out, top_hit_string, precursor_mass_list = ef.find_formulas(mass_list, ppm=ppm_max, polarisation=mode.upper())
        
        # replace all None with NaN
        for i in range(len(top_hit_string)):
            if top_hit_string[i] is None:
                top_hit_string[i] = np.NaN        

        # set the results back into the dataframe        
        self.ms1['annotation'] = top_hit_string

        
    # def annotate_with_name(self, peaklist_file, ppm_max=5):  
        
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