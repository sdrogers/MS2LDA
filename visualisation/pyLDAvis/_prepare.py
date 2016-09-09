"""
pyLDAvis Prepare
===============
Main transformation functions for preparing LDAdata to the visualization's data structures
NOT the same as the original pyLDAvis -- modified for MS2LDAvis!
"""

from collections import namedtuple
import json
import numpy as np
import pandas as pd
from .utils import NumPyEncoder

def __num_dist_rows__(array, ndigits=2):
    return int(pd.DataFrame(array).sum(axis=1).map(lambda x: round(x, ndigits)).sum())


class ValidationError(ValueError):
    pass


def _input_check(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency):
    ttds = topic_term_dists.shape
    dtds = doc_topic_dists.shape
    errors = []
    def err(msg):
        errors.append(msg)

    if dtds[1] != ttds[0]:
        err('Number of rows of topic_term_dists does not match number of columns of doc_topic_dists; both should be equal to the number of topics in the model.')

    if len(doc_lengths) != dtds[0]:
        err('Length of doc_lengths not equal to the number of rows in doc_topic_dists; both should be equal to the number of documents in the data.')

    W = len(vocab)
    if ttds[1] != W:
        err('Number of terms in vocabulary does not match the number of columns of topic_term_dists (where each row of topic_term_dists is a probability distribution of terms for a given topic).')
    if len(term_frequency) != W:
        err('Length of term_frequency not equal to the number of terms in the vocabulary (len of vocab).')

    if __num_dist_rows__(topic_term_dists) != ttds[0]:
        err('Not all rows (distributions) in topic_term_dists sum to 1.')

    if __num_dist_rows__(doc_topic_dists) != dtds[0]:
        err('Not all rows (distributions) in doc_topic_dists sum to 1.')

    if len(errors) > 0:
        return errors


def _input_validate(*args):
    res = _input_check(*args)
    if res:
        raise ValidationError('\n' + '\n'.join([' * ' + s for s in res]))

def _df_with_names(data, index_name, columns_name):
    if type(data) == pd.DataFrame:
        # we want our index to be numbered
        df = pd.DataFrame(data.values)
    else:
        df = pd.DataFrame(data)
    df.index.name = index_name
    df.columns.name = columns_name
    return df


def _series_with_name(data, name):
    if type(data) == pd.Series:
        data.name = name
        # ensures a numeric index
        return data.reset_index()[name]
    else:
        return pd.Series(data, name=name)


def _topic_coordinates(topic_term_dists, topic_proportion, topic_coords):
    K = topic_term_dists.shape[0]
    assert(K==len(topic_coords))
    xs = [coor[0] for coor in topic_coords]
    ys = [coor[1] for coor in topic_coords]
    mds_df = pd.DataFrame({'x': xs, 'y': ys, 'topics': range(1, K + 1), \
                                  'cluster': 1, 'Freq': topic_proportion * 100})

    return mds_df


def _find_relevance(log_ttd, log_lift, R, lambda_):
    relevance = lambda_ * log_ttd + (1 - lambda_) * log_lift
    return relevance.T.apply(lambda s: s.order(ascending=False).index).head(R)


def _find_relevance_chunks(log_ttd, log_lift, R, lambda_seq):
    return pd.concat([_find_relevance(log_ttd, log_lift, R, l) for l in lambda_seq])


def _topic_info(topic_term_dists, topic_proportion, term_frequency, term_topic_freq,
                topic_wordfreq, topic_ms1_count,
                vocab, lambda_step, R, n_jobs):
    # marginal distribution over terms (width of blue bars)
    term_proportion = term_frequency / term_frequency.sum()

    # compute the distinctiveness and saliency of the terms:
    # this determines the R terms that are displayed when no topic is selected
    topic_given_term = topic_term_dists / topic_term_dists.sum()
    kernel = (topic_given_term * np.log((topic_given_term.T / topic_proportion).T))
    distinctiveness = kernel.sum()
    saliency = term_proportion * distinctiveness

    # Order the terms for the "default" view by decreasing saliency:
    default_term_info  = pd.DataFrame({'saliency': saliency, 'Term': vocab, \
                                                  'Freq': term_frequency, 'Total': term_frequency, \
                                                  'Category': 'Default'}). \
        sort('saliency', ascending=False). \
        head(R).drop('saliency', 1)
    ranks = np.arange(R, 0, -1)
    default_term_info['logprob'] = default_term_info['loglift'] = ranks

    ## compute relevance and top terms for each topic
    log_lift = np.log(topic_term_dists / term_proportion)
    log_ttd = np.log(topic_term_dists)

    lambda_step = 0.1
    lambda_seq = np.arange(0, 1 + lambda_step, lambda_step)

    def topic_top_term_df(tup):
        new_topic_id, (original_topic_id, topic_terms) = tup
        term_ix = topic_terms.unique()

        label = vocab[term_ix]
        freq = term_topic_freq.loc[original_topic_id, term_ix]

        topic_label_count = topic_wordfreq[original_topic_id]
        label_count = freq.copy()
        for freq_idx, freq_val in freq.iteritems():
            word = label[freq_idx]
            if word in topic_label_count:
                label_count[freq_idx] = topic_label_count[word]
            else:
                label_count[freq_idx] = 0

        return pd.DataFrame({'Term': label, \
                                    'Freq': freq, \
                                    'Total': term_frequency[term_ix], \
                                    'logprob': log_ttd.loc[original_topic_id, term_ix].round(4), \
                                    'loglift': log_lift.loc[original_topic_id, term_ix].round(4), \
                                    'prob': topic_term_dists.loc[original_topic_id, term_ix], \
                                    'label_count': label_count, \
                                    'ms1_count': topic_ms1_count[original_topic_id], \
                                    'Category': 'Topic%d' % new_topic_id})

    top_terms = _find_relevance_chunks(log_ttd, log_lift, R, lambda_seq)
    topic_dfs = map(topic_top_term_df, enumerate(top_terms.T.iterrows(), 1))
    return pd.concat([default_term_info] + list(topic_dfs))


def _token_table(topic_info, term_topic_freq, vocab, term_frequency):
    # last, to compute the areas of the circles when a term is highlighted
    # we must gather all unique terms that could show up (for every combination
    # of topic and value of lambda) and compute its distribution over topics.

    # term-topic frequency table of unique terms across all topics and all values of lambda
    term_ix = topic_info.index.unique()
    term_ix.sort()
    top_topic_terms_freq = term_topic_freq[term_ix]
    # use the new ordering for the topics
    K = len(term_topic_freq)
    top_topic_terms_freq.index = range(1, K + 1)
    top_topic_terms_freq.index.name = 'Topic'

    # we filter to Freq >= 0.5 to avoid sending too much data to the browser
    token_table = pd.DataFrame({'Freq': top_topic_terms_freq.unstack()}). \
                      reset_index().set_index('term'). \
                      query('Freq >= 0.5')

    token_table['Freq'] = token_table['Freq'].round()
    token_table['Term'] = vocab[token_table.index.values].values
    # Normalize token frequencies:
    token_table['Freq'] = token_table.Freq / term_frequency[token_table.index]
    return token_table.sort(['Term', 'Topic'])


def _term_topic_freq(topic_term_dists, topic_freq, term_frequency):
    term_topic_freq = (topic_term_dists.T  * topic_freq).T
    # adjust to match term frequencies exactly (get rid of rounding error)
    err = term_frequency / term_topic_freq.sum()
    return term_topic_freq * err


def prepare(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency, topic_ranking, topic_coordinates, plot_opts, \
                lambda_step, lambda_min, lambda_max, th_topic_word, th_doc_topic, topic_wordfreq, topic_ms1_count, \
                topic_annotation, R=30, n_jobs=-1):
    """Transforms the topic model distributions and related corpus data into
    the data structures needed for the visualization.

     Parameters
     ----------
     topic_term_dists : array-like, shape (`n_topics`, `n_terms`)
          Matrix of topic-term probabilities. Where `n_terms` is `len(vocab)`.
     doc_topic_dists : array-like, shape (`n_docs`, `n_topics`)
          Matrix of document-topic probabilities.
     doc_lengths : array-like, shape `n_docs`
          The length of each document, i.e. the number of words in each document.
          The order of the numbers should be consistent with the ordering of the
          docs in `doc_topic_dists`.
     vocab : array-like, shape `n_terms`
          List of all the words in the corpus used to train the model.
     term_frequency : array-like, shape `n_terms`
          The count of each particular term over the entire corpus. The ordering
          of these counts should correspond with `vocab` and `topic_term_dists`.
     R : int
          The number of terms to display in the barcharts of the visualization.
          Default is 30. Recommended to be roughly between 10 and 50.
     lambda_step : float, between 0 and 1
          Determines the interstep distance in the grid of lambda values over
          which to iterate when computing relevance.
          Default is 0.01. Recommended to be between 0.01 and 0.1.
     n_jobs: int
          The number of cores to be used to do the computations. The regular
          joblib conventions are followed so `-1`, which is the default, will
          use all cores.
     plot_opts : dict, with keys 'xlab' and `ylab`
          Dictionary of plotting options, right now only used for the axis labels.

     Returns
     -------
     prepared_data : PreparedData
          A named tuple containing all the data structures required to create
          the visualization. To be passed on to functions like :func:`display`.

     Notes
     -----
     This implements the method of `Sievert, C. and Shirley, K. (2014):
     LDAvis: A Method for Visualizing and Interpreting Topics, ACL Workshop on
     Interactive Language Learning, Visualization, and Interfaces.`

     http://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf

     See Also
     --------
     :func:`save_json`: save json representation of a figure to file
     :func:`save_html` : save html representation of a figure to file
     :func:`show` : launch a local server and show a figure in a browser
     :func:`display` : embed figure within the IPython notebook
     :func:`enable_notebook` : automatically embed visualizations in IPython notebook
    """
    topic_term_dists = _df_with_names(topic_term_dists, 'topic', 'term')
    doc_topic_dists  = _df_with_names(doc_topic_dists, 'doc', 'topic')
    term_frequency    = _series_with_name(term_frequency, 'term_frequency')
    doc_lengths        = _series_with_name(doc_lengths, 'doc_length')
    vocab                = _series_with_name(vocab, 'vocab')
    _input_validate(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency)
    R = min(R, len(vocab))

    topic_freq         = (doc_topic_dists.T * doc_lengths).T.sum()
    topic_proportion = (topic_freq / topic_freq.sum()).order(ascending=False)
#     topic_order        = topic_proportion.index
    # reorder all data based on new ordering of topics
#     topic_freq         = topic_freq[topic_order]
#     topic_term_dists = topic_term_dists.ix[topic_order]
#     doc_topic_dists  = doc_topic_dists[topic_order]


    # token counts for each term-topic combination (widths of red bars)
    term_topic_freq     = _term_topic_freq(topic_term_dists, topic_freq, term_frequency)
    topic_info            = _topic_info(topic_term_dists, topic_proportion, term_frequency, term_topic_freq,
                                        topic_wordfreq, topic_ms1_count,
                                        vocab, lambda_step, R, n_jobs)
    token_table          = _token_table(topic_info, term_topic_freq, vocab, term_frequency)
    topic_coordinates = _topic_coordinates(topic_term_dists, topic_proportion, topic_coordinates)

    # ignore topic_order completely and leave the ordering of the topics as it is
#     client_topic_order = [x + 1 for x in topic_order]
    K = topic_term_dists.shape[0]
    client_topic_order = range(K)

    # create a list of the annotation for each topic
    topic_annotation_list = []
    for topic_id in topic_ranking[:, 0]:
        annot = None
        if topic_id in topic_annotation:
            annot = topic_annotation[topic_id]
        topic_annotation_list.append(annot)

    # pass the topic h-indices for displaying on the front-end later
    topic_ranking = pd.DataFrame({'topic_id': topic_ranking[:, 0],
                                  'rank': topic_ranking[:, 1],
                                  'degree': topic_ranking[:, 2],
                                  'annotation': topic_annotation_list}).sort('rank', ascending=False)

    return PreparedData(topic_coordinates, topic_info, token_table, R, lambda_step, lambda_min, lambda_max, plot_opts, \
                        client_topic_order, topic_ranking, th_topic_word, th_doc_topic)

class PreparedData(namedtuple('PreparedData', ['topic_coordinates', 'topic_info', 'token_table',\
                                                              'R', 'lambda_step', 'lambda_min', 'lambda_max', \
                                                              'plot_opts', 'topic_order', 'topic_ranking', \
                                                              'th_topic_word', 'th_doc_topic'])):
    def to_dict(self):
        return {'mdsDat': self.topic_coordinates.to_dict(orient='list'),
                    'tinfo': self.topic_info.to_dict(orient='list'),
                    'token.table': self.token_table.to_dict(orient='list'),
                    'R': self.R,
                    'lambda.step': self.lambda_step,
                    'lambda.min': self.lambda_min,
                    'lambda.max': self.lambda_max,
                    'plot.opts': self.plot_opts,
                    'topic.order': self.topic_order,
                    'topic.ranking': self.topic_ranking.to_dict(orient='list'),
                    'th_topic_word': self.th_topic_word,
                    'th_doc_topic': self.th_doc_topic}

    def to_json(self):
        return json.dumps(self.to_dict(), cls=NumPyEncoder)
