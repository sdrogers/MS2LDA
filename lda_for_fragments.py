import os
import re
import sys
import timeit

from numpy.random.mtrand import RandomState
from pandas.core.frame import DataFrame
from scipy.sparse import coo_matrix

from visualisation.pylab.lda_for_fragments_viz import Ms2Lda_Viz
from lda_cgs import CollapseGibbsLda
import numpy as np
import pandas as pd


try:
    from lda import LDA
except Exception:
    pass



class Ms2Lda(object):
    
    def __init__(self, fragment_filename, neutral_loss_filename, mzdiff_filename, 
                 ms1_filename, ms2_filename, relative_intensity=True):
        
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

        # discretise the fragment and neutral loss intensities values by converting it to 0 .. 100
        # values are already normalised from 0 .. 1 during feature extraction
        self.fragment_data *= 100
        self.neutral_loss_data *= 100
        self.data = self.fragment_data.append(self.neutral_loss_data)
    
        # then scale mzdiff counts from 0 .. 100 too, and append it to data    
        if self.mzdiff_data is not None:
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
                use_own_model=False, use_native=True, previous_model=None):    
                        
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
        
    def plot_lda_fragments(self, consistency=0.50, sort_by="h_index", selected_topics=None):
        plotter = Ms2Lda_Viz(self.model, self.ms1, self.ms2, self.docdf, self.topicdfs)
        plotter.plot_lda_fragments(consistency=0.50, sort_by=sort_by, selected_topics=selected_topics)         
        
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
        n_topics = 125
    n_samples = 20
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
#     ms2lda.write_results('beer3pos')
#     ms2lda.plot_lda_fragments(consistency=0.50)

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