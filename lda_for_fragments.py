import os
import re
import sys
import timeit

from pandas.core.frame import DataFrame
from scipy.sparse import coo_matrix

from visualisation.pylab.lda_for_fragments_viz import Ms2Lda_Viz
from lda_cgs import CollapseGibbsLda
import numpy as np
import pandas as pd
import pylab as plt

class Ms2Lda(object):
    
    def __init__(self, df, vocab, ms1, ms2, EPSILON=0.05):
        self.df = df
        self.vocab = vocab
        self.ms1 = ms1
        self.ms2 = ms2
        self.EPSILON = EPSILON
        
    @classmethod
    def lcms_data_from_R(cls, fragment_filename, neutral_loss_filename, mzdiff_filename, 
                 ms1_filename, ms2_filename, vocab_type=1):

        fragment_data = None
        neutral_loss_data = None
        mzdiff_data = None

        # load all the input files        
        if fragment_filename is not None:
            fragment_data = pd.read_csv(fragment_filename, index_col=0)
        if neutral_loss_filename is not None:
            neutral_loss_data = pd.read_csv(neutral_loss_filename, index_col=0)
        if mzdiff_filename is not None:
            mzdiff_data = pd.read_csv(mzdiff_filename, index_col=0)
        
        ms1 = pd.read_csv(ms1_filename, index_col=0)
        ms2 = pd.read_csv(ms2_filename, index_col=0)
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
        df = DataFrame(npdata)
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
        this_instance = cls(df, vocab, ms1, ms2)
        return this_instance           
        
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
        this_instance = cls(df, vocab, ms1, ms2)
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
        topic_word = self.model.topic_word_
        with open(outfile,'w') as f:
            
            counter = 0
            for i, topic_dist in enumerate(topic_word):

                ordering = np.argsort(topic_dist)
                vocab = self.df.columns.values                
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
        masses = np.array(self.df.transpose().index)
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
        doc_names = np.array(self.df.index)
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
        plotter = Ms2Lda_Viz(self.model, self.ms1, self.ms2, self.docdf, self.topicdf)
        return plotter.rank_topics(sort_by=sort_by, selected_topics=selected_topics, top_N=top_N)
        
    def plot_lda_fragments(self, consistency=0.50, sort_by="h_index", selected_topics=None, interactive=False):
        plotter = Ms2Lda_Viz(self.model, self.ms1, self.ms2, self.docdf, self.topicdf)
        plotter.plot_lda_fragments(consistency=consistency, sort_by=sort_by, 
                                   selected_topics=selected_topics, interactive=interactive)
        if interactive:
            self.model.visualise(topic_plotter=plotter)
            
    def print_topic_words(self):
        self.model.print_topic_words(self.EPSILON)
        
    def plot_posterior_alpha(self):
        posterior_alpha = self.model.posterior_alpha
        posterior_alpha = posterior_alpha / np.sum(posterior_alpha)
        ind = range(len(posterior_alpha))
        plt.bar(ind, posterior_alpha, 2)

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

    n_samples = 20
    n_burn = 0
    n_thin = 1
    alpha = 50.0/n_topics
    beta = 0.1

    # train on beer3pos
    
    fragment_filename = 'input/relative_intensities/Beer_3_T10_POS_fragments_rel.csv'
    neutral_loss_filename = 'input/relative_intensities/Beer_3_T10_POS_losses_rel.csv'
    mzdiff_filename = None    
    ms1_filename = 'input/relative_intensities/Beer_3_T10_POS_ms1_rel.csv'
    ms2_filename = 'input/relative_intensities/Beer_3_T10_POS_ms2_rel.csv'
  
    ms2lda = Ms2Lda.lcms_data_from_R(fragment_filename, neutral_loss_filename, mzdiff_filename, 
                                     ms1_filename, ms2_filename)    
    ms2lda.run_lda(n_topics, n_samples, n_burn, n_thin, alpha, beta)
    ms2lda.write_results('beer3pos')
    ms2lda.model.print_topic_words()    
    ms2lda.plot_lda_fragments(consistency=0.50, sort_by="h_index", interactive=False)

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