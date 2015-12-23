"""
Visualisation methods for the LDA results
"""

import SimpleHTTPServer
import SocketServer
import json
import sys

from networkx.readwrite import json_graph
from pandas.core.frame import DataFrame

import networkx as nx
import numpy as np
import pandas as pd
import pylab as plt
import csv
import matplotlib.patches as mpatches
from matplotlib.colors import rgb2hex
import math

# this should match the same constant defined in graph.html
TOPIC_NAME = "motif" 

def _get_docname(row_index):
    tokens = row_index.split('_')
    docname = "doc_"+ str(tokens[0]) + "_" + str(tokens[1])
    peakid = str(tokens[2])
    return docname, peakid

def _get_topicname(col_index):
    topic = TOPIC_NAME + "_" + str(col_index)
    return topic
    
def get_json_from_docdf(docdf, to_highlight, threshold, selected_motifs=None):

    if to_highlight is None:
        to_highlight_labels = []
        to_highlight_colours = {}
    else:
        to_highlight_labels = [x[0] for x in to_highlight]
        to_highlight_colours = {}
        for x in to_highlight:
            label = x[0]
            colour = x[1]
            to_highlight_colours[label] = colour

    G = nx.Graph()
    node_names = set()
    peakid_map = {}
    for row_index in docdf.index:
        docname, peakid = _get_docname(row_index)
        node_names.add(docname)
        peakid_map[docname] = peakid
    for col_index in docdf.columns:
        topic = _get_topicname(col_index)
        node_names.add(topic)

    nodes = {}
    node_id = 0
    for n in node_names:
        nodes[n] = node_id
        node_id += 1
        
    for row_index, row_value in docdf.iterrows():    
        for col_index, col_value in row_value.iteritems():
            if col_value > 0:
                docname, peakid = _get_docname(row_index)
                topic = _get_topicname(col_index)
                docid = nodes[docname]
                topicid = nodes[topic]
                weight = col_value
                if weight > 0:
                    G.add_edge(docid, topicid, weight=weight)

    for n in node_names:
        node_id = nodes[n]
        if node_id in G:

            # always insert all documents
            if n.startswith('doc'):
                node_group = 1
                node_size = 10
                node_score = 0
                node_type = "square"
                special = False
                pid = peakid_map[n]
                n_pid = 'doc_' + pid
                if n_pid in to_highlight_labels:
                    node_size = 30
                    special = True
                    highlight_colour = to_highlight_colours[n_pid]
                    G.add_node(node_id, name=n, group=node_group, in_degree=0, size=node_size, score=node_score, 
                               type=node_type, special=special, highlight_colour=highlight_colour, peakid=pid)
                else:
                    G.add_node(node_id, name=n, group=node_group, in_degree=0, size=node_size, score=node_score, 
                               type=node_type, special=special, peakid=pid)

            # for topics, insert only those whose in-degree is above threshold
            elif n.startswith(TOPIC_NAME):
                node_group = 2
                node_size = 60
                node_score = 1
                node_type = "circle"
                special = False
                in_degree = G.degree(node_id)

                included = True
                if in_degree >= threshold:
                    if n in to_highlight_labels:
                        special = True
                        highlight_colour = to_highlight_colours[n]                        
                        G.add_node(node_id, name=n, group=node_group, in_degree=in_degree, size=in_degree*5, score=node_score, 
                                   type=node_type, special=special, highlight_colour=highlight_colour)                        
                    else:
                        G.add_node(node_id, name=n, group=node_group, in_degree=in_degree, size=in_degree*5, score=node_score, 
                                   type=node_type, special=special)
#                     print(str(node_id) + ", " + n + " degree=" + str(in_degree) + " added")        
                else:
                    G.remove_node(node_id)
#                     print(str(node_id) + ", " + n + " degree=" + str(in_degree) + " removed")   
                    included = True                                   

                if selected_motifs is not None and n not in selected_motifs:
                    G.remove_node(node_id)
                    included = False

#                 if included:
#                     print(str(node_id) + ", " + n + " degree=" + str(in_degree) + " added")        
#                 else:
#                     print(str(node_id) + ", " + n + " degree=" + str(in_degree) + " removed")   


    # final cleanup, delete all unconnected documents
    unconnected = []
    for n in node_names:
        node_id = nodes[n]
        if node_id in G:
            degree = G.degree(node_id) 
            if n.startswith('doc') and degree == 0:
                unconnected.append(node_id)
    G.remove_nodes_from(unconnected)    

#     print("Total nodes = " + str(G.number_of_nodes()))
#     print("Total edges = " + str(G.number_of_edges()))

    json_out = json_graph.node_link_data(G) # node-link format to serialize
    return json_out, G

def get_json_from_topicdf(topicdf):

    G = nx.Graph()
    node_names = set()
    for row_index in topicdf.index:
        node_names.add(row_index)
    for col_index in topicdf.columns:
        node_names.add(col_index)

    nodes = {}
    node_id = 0
    for n in node_names:
        nodes[n] = node_id
        node_id += 1
        
    for row_index, row_value in topicdf.iterrows():    
        for col_index, col_value in row_value.iteritems():
            if col_value > 0:
                term = row_index
                topic = col_index
                termid = nodes[term]
                topicid = nodes[topic]
                weight = col_value
                if weight > 0:
                    G.add_edge(termid, topicid, weight=weight)

    for n in node_names:
        node_id = nodes[n]
        G.add_node(node_id, name=n)

    print("Total nodes = " + str(G.number_of_nodes()))
    print("Total edges = " + str(G.number_of_edges()))

    json_out = json_graph.node_link_data(G) # node-link format to serialize
    return json_out, G

def export_docdf_to_networkx(infile):
    """ Exports docdf to networkx """

    print("Loading " + infile)
    docdf = pd.read_csv(infile, index_col=0)
    d = get_json_from_docdf(docdf)
    json.dump(d, open('test.json','w'))
    print('Wrote node-link JSON data to test.json') 
    
    PORT = 1234 
    Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
    httpd = SocketServer.TCPServer(("", PORT), Handler)
    print("serving at port " + str(PORT))
    httpd.serve_forever()        

def get_network_graph(ms2lda, motifs_of_interest, verbose=True):

    G, json_data = ms2lda.get_network_graph(to_highlight=None, degree_filter=0)
    motifs_of_interest = ['motif_' + str(id) for id in motifs_of_interest]

    # 1. keep only the motifs in the list
    remove_count = 0
    nodes = G.nodes(data=True)
    for node_id, node_data in nodes:
        # 1 for doc, 2 for motif
        if node_data['group'] == 2 and node_data['name'] not in motifs_of_interest: 
            remove_count += 1
            G.remove_node(node_id)
    if verbose:
        print "Removed %d motifs from the graph because they're not in the list" % remove_count

    # 2. keep only motifs having shared nodes with other motifs
    removed_motifs = []
    nodes = G.nodes(data=True)    
    for node_id, node_data in nodes:
        if node_data['group'] == 2:
            # check if any doc in this motif is shared with another motif (degree > 1)
            neighbours = G.neighbors(node_id)
            share = False
            for nb in neighbours:
                deg = G.degree(nb)
                if deg > 1:
                    share = True
            # if not then delete from the graph too
            if not share:
                removed_motifs.append(node_data['name'])
                G.remove_node(node_id)
    if verbose:
        print "Removed %s from the graph because they don't share documents with other motifs in the list" % removed_motifs
    
    # 3. delete all unconnected nodes from the graph
    unconnected = []
    nodes = G.nodes(data=True)    
    for node_id, node_data in nodes:
        if G.degree(node_id) == 0:
            unconnected.append(node_id)
    G.remove_nodes_from(unconnected)   
    if verbose:
        print "Removed %d unconnected documents from the graph" % len(unconnected)
                    
    return G
    
def plot_bipartite(G, min_degree, fig_width=10, fig_height=20, spacing_left=1, spacing_right=2):
        
    # extract subgraph of docs connected to at least min_degree motifs
    doc_nodes_to_keep = set()
    motif_nodes_to_keep = set()    
    nodes = G.nodes(data=True)    
    for node_id, node_data in nodes:
        # group == 1 is a doc, 2 is a motif
        if node_data['group'] == 1 and G.degree(node_id) >= min_degree:
            neighbours = G.neighbors(node_id)
            doc_nodes_to_keep.add(node_id)
            motif_nodes_to_keep.update(neighbours)
    to_keep = doc_nodes_to_keep | motif_nodes_to_keep # set union
    SG = nx.Graph(G.subgraph(to_keep))

    # make bipartite layout, put doc nodes on left, motif nodes on right
    pos = dict()
    pos.update( (n, (1, i*spacing_left)) for i, n in enumerate(doc_nodes_to_keep) )
    pos.update( (n, (2, i*spacing_right)) for i, n in enumerate(motif_nodes_to_keep) )

    # for labelling purpose
    motif_singleton = {}
    for n in motif_nodes_to_keep:
        children = G.neighbors(n) # get the children of this motif
        degree_dict = G.degree(nbunch=children) # get the degrees of children
        # count how many children have degree == 1
        children_degrees = [degree_dict[c] for c in degree_dict]
        count_singleton = sum(child_deg == 1 for child_deg in children_degrees)        
        motif_singleton[n] = count_singleton
        
    # set the node and edge labels
    doc_labels = {}
    motif_labels = {}
    doc_motifs = {} # used for the fragmentation spectra plot
    all_motifs = set()
    for node_id, node_data in SG.nodes(data=True):
        if node_data['group'] == 2: # is a motif
            motif_labels[node_id] = "%s (+%d)" % (node_data['name'], motif_singleton[node_id])
        elif node_data['group'] == 1: # is a doc
            pid = int(node_data['peakid'])
            doc_labels[node_id] = 'pid_%d' % pid
            parent_motifs = set()
            for neighbour_id in SG.neighbors(node_id):
                motif_name = SG.node[neighbour_id]['name']
                _, motif_id = motif_name.split('_')
                parent_motifs.add(int(motif_id))
            doc_motifs[pid] = parent_motifs
            all_motifs.update(parent_motifs)
            
    # plot the bipartite graph
    plt.figure(figsize=(fig_width, fig_height))
    plt.axis('off')
    fig = plt.figure(1)
    nx.draw_networkx_nodes(SG, pos, alpha=0.25)
    nx.draw_networkx_edges(SG, pos, alpha=0.25)
    _ = nx.draw_networkx_labels(SG, pos, doc_labels, font_size=10)
    _ = nx.draw_networkx_labels(SG, pos, motif_labels, font_size=16)
    ymax = max(len(doc_labels)*spacing_left, len(motif_labels)*spacing_right)
    _ = plt.ylim([-1, ymax])
    _ = plt.title('MS1 peaks connected to at least %d motifs' % min_degree)
    
    # assign index to each M2M
    i = 0
    motif_idx = {}
    for key in all_motifs:
        motif_idx[key] = i
        i += 1    
    
    return doc_nodes_to_keep, doc_motifs, motif_idx    
    
def get_word_motif(word, doc_motifs, word_map):
    word_motif = np.NaN
    if word in word_map:
        to_check = word_map[word]
        same_motifs = list(to_check.intersection(doc_motifs))
        if len(same_motifs) > 0:
            word_motif = same_motifs[0]
    return word_motif

def plot_subgraph(G, m2m_list, ms1_peakids_to_highlight, motif_idx, colour_map, save_to=None):
    
    # parameters for the M2M nodes
    motif_node_alpha = 0.75
    motif_node_size = 2000
    
    # parameters for the ms1_peakids_to_highlight
    highlight_node_colour = 'gray'
    highlight_node_alpha = 0.75
    highlight_node_size = 600
    
    # parameters for other nodes
    other_nodes_colour = 'lightgray'
    other_nodes_alpha = 0.20
    other_nodes_size = 600

    # label parameters
    label_colour = 'black'
    label_background_colour = 'gray'
    label_alpha = 0.50
    label_fontsize = 16
    
    # other parameters
    fig_width = 15
    fig_height = 15
    edge_alpha = 0.25    
        
    highlight_docs = set()
    other_docs = set()
    motif_nodes = set()    
    motif_colours = []
    doc_labels = {}
    motif_labels = {}
    for node_id, node_data in G.nodes(data=True):
        
        # group == 1 is a doc, 2 is a motif
        if node_data['group'] == 1 and int(node_data['peakid']) in ms1_peakids_to_highlight:
            highlight_docs.add(node_id)
            doc_labels[node_id] = node_data['peakid']

        elif node_data['group'] == 2:
            motif_name = node_data['name']
            _, motif_id = motif_name.split('_')
            motif_id = int(motif_id)
            if motif_id in m2m_list:
                motif_nodes.add(node_id)
                neighbours = G.neighbors(node_id)
                other_docs.update(neighbours)
                node_colour = colour_map.to_rgba(motif_idx[motif_id])
                node_colour = rgb2hex(node_colour)
                motif_colours.append(node_colour)
                motif_labels[node_id] = 'M2M_%d' % motif_id

    other_docs = other_docs - highlight_docs
    to_keep = motif_nodes | highlight_docs | other_docs
    SG = nx.Graph(G.subgraph(to_keep))
    for u, v, d in SG.edges(data=True):
        d['weight'] *= 100

    plt.figure(figsize=(fig_width, fig_height), dpi=900)
    plt.axis('off')
    fig = plt.figure(1)
    
    pos = nx.spring_layout(SG, k=0.25, iterations=50)
    nx.draw_networkx_nodes(SG, pos, nodelist=other_docs, 
                           alpha=other_nodes_alpha, node_color=other_nodes_colour, 
                           node_size=other_nodes_size)
    nx.draw_networkx_nodes(SG, pos, nodelist=highlight_docs, 
                           alpha=highlight_node_alpha, node_color=highlight_node_colour, 
                           node_size=highlight_node_size)
    nx.draw_networkx_nodes(SG, pos, nodelist=motif_nodes, 
                           alpha=motif_node_alpha, node_color=motif_colours,
                           node_size=motif_node_size)
    nx.draw_networkx_edges(SG, pos, alpha=edge_alpha)    
    nx.draw_networkx_labels(SG, pos, doc_labels, font_size=10)
    nx.draw_networkx_labels(SG, pos, motif_labels, font_size=16)    
    
#     for node_id in node_labels:
#         x, y = pos[node_id]
#         label = node_labels[node_id]
#         y_offset = -0.3
#         if node_id in motif_nodes:
#             y_offset += -0.5
#         plt.text(x,y+y_offset, s=node_labels[node_id], 
#                  color=label_colour, fontsize=label_fontsize,
#                  bbox=dict(facecolor=label_background_colour, alpha=label_alpha), 
#                  horizontalalignment='center') 
        
    if save_to is not None:
        print "Figure saved to %s" % save_to
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()        
    
    return SG
    
def plot_fragmentation_spectrum(ms2_df, ms1_row, motif_colour, motif_idx, ms1_label=None, save_to=None, xlim_upper=300):
    
    # make sure that the fragment and loss words got plotted first in the loop below
    ms2_df = ms2_df.fillna(value=np.NaN)
    ms2_df.sort_values(['fragment_motif', 'loss_motif'], ascending=True, inplace=True, na_position='last')
    
    # set figure and font sizes
    plt.figure(figsize=(20, 10), dpi=900)
    ax = plt.subplot(111)
    large_font_size = 32
    small_font_size = 24
    linewidth = 4

    # the ms1 info
    ms1_peakid = ms1_row['peakID'].values[0]
    ms1_mz = ms1_row['mz'].values[0]    
    ms1_rt = ms1_row['rt'].values[0]
    ms1_intensity = ms1_row['intensity'].values[0]
    ms1_annotation = ms1_row['annotation'].values[0]    
    
    # hacky
    neutral_loss_positions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    neutral_loss_count = 0

    # plot each MS2 peak    
    for row_index, row in ms2_df.iterrows():

        mz = row['ms2_mz']
        intensity = row['ms2_intensity']    
        frag_m2m = row['fragment_motif']
        loss_m2m = row['loss_motif']
        frag_word = row['fragment_word']
        loss_word = row['loss_word']
        
        if not np.isnan(loss_m2m):

            # draw neutral loss arrow
            word_colour = get_colour(loss_m2m, motif_idx, motif_colour)
            arrow_x1 = ms1_mz
            arrow_y1 = neutral_loss_positions[neutral_loss_count]
            arrow_x2 = (mz-ms1_mz)+5
            arrow_y2 = 0
            neutral_loss_count += 1
            plt.arrow(arrow_x1, arrow_y1, arrow_x2, arrow_y2, head_width=0.05, head_length=4.0, width=0.005, fc=word_colour, ec=word_colour)

            # draw neutral loss label
            text_x = mz+(ms1_mz-mz)/4
            text_y = arrow_y1+0.025
            t = ax.text(text_x, text_y, loss_word.split('_')[1], ha="left", va="center", rotation=0,
                        size=small_font_size, color=word_colour)

        # draw the fragment
        if not np.isnan(frag_m2m):
            word_colour = get_colour(frag_m2m, motif_idx, motif_colour)            
            if intensity > 0.2:
                ax.text(mz, intensity+0.025, frag_word.split('_')[1], ha="center", va="center", 
                        size=small_font_size, color=word_colour)
        else:
            word_colour = 'lightgray'
        plt.plot((mz, mz), (0, intensity), linewidth=linewidth, color=word_colour)

    # plot the ms1 peak too
    plt.plot((ms1_mz, ms1_mz), (0, 0.25), linewidth=linewidth, color='k')
    plt_label = "MS1 mz=%.4f RT=%.2f" % (ms1_mz, ms1_rt)
    ax.text(ms1_mz+2, 0.38, plt_label, ha="left", va="center", 
            size=small_font_size, color='k', rotation=90)

    # set title, axes labels and ranges
    if ms1_label is not None and ms1_peakid in ms1_label:
        ms1_annotation += "; " + ms1_label[ms1_peakid]
    title = "Fragmentation spectrum for MS1 peakID %d (%s)" % (ms1_peakid, ms1_annotation)    
    plt.title(title)
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')
    plt.xlim([0, ms1_mz + 100])    
    plt.ylim([0, 1.1])

    # plot legend
    fragment_m2m = ms2_df['fragment_motif'].values
    fragment_m2m = set(fragment_m2m[~np.isnan(fragment_m2m)].tolist())
    loss_m2m = ms2_df['loss_motif'].values
    loss_m2m = set(loss_m2m[~np.isnan(loss_m2m)].tolist())
    m2m_used = fragment_m2m | loss_m2m
    m2m_patches = []
    for m2m in m2m_used:
        m2m_colour = get_colour(m2m, motif_idx, motif_colour)        
        m2m_patch = mpatches.Patch(color=m2m_colour, label='M2M_%d' % m2m)
        m2m_patches.append(m2m_patch)
    ax.legend(handles=m2m_patches, loc='upper right', 
              ncol=1, fancybox=True, shadow=True, prop={'size': large_font_size})        

    # increase fontsize
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(large_font_size)    

    # save figure
    if save_to is not None:
        print "Figure saved to %s" % save_to
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()
    
def get_colour(m2m, motif_idx, motif_colour):
    try:
        word_colour = motif_colour.to_rgba(motif_idx[m2m])
    except AttributeError:
        word_colour = motif_colour[m2m]
    return word_colour    
        
def print_report(ms2lda, G, peak_id, motif_annotation, motif_words, motif_colour, motif_idx, word_map, 
                 ms1_label=None, save_to=None, xlim_upper=300, contributions=None, plotting_func=None):
    
    doc_motifs = {}
    for node_id, node_data in G.nodes(data=True):
        if node_data['group'] == 1: # is a doc
            pid = int(node_data['peakid'])
            parent_motifs = set()
            for neighbour_id in G.neighbors(node_id):
                motif_name = G.node[neighbour_id]['name']
                _, motif_id = motif_name.split('_')
                parent_motifs.add(int(motif_id))
            doc_motifs[pid] = parent_motifs    
            
    # get the ms1 info
    ms1_rows = ms2lda.ms1.loc[ms2lda.ms1['peakID'] == peak_id]
    first_row = ms1_rows.head(1)
    
    try:
        ms1_motifs = doc_motifs[peak_id]    
    except KeyError:
        print " - No M2M for this MS1 peak at the specified thresholding levels"
        print
        return None
    
    for m2m in ms1_motifs:
        try:
            m2m_annot = motif_annotation[m2m]
        except KeyError:
            m2m_annot = '?'
        print " - M2M_%s\t: %s" % (m2m, m2m_annot)
        print "\t\t  %s" % (motif_words[m2m])
    print

    # get the ms2 info
    ms2_rows = ms2lda.ms2.loc[ms2lda.ms2['MSnParentPeakID'] == peak_id]
    ms2_mz = ms2_rows['mz'].values
    ms2_intensity = ms2_rows['intensity'].values
    ms2_fragment_words = ms2_rows['fragment_bin_id'].values
    ms2_loss_words = ms2_rows['loss_bin_id'].values
    if 'annotation' in ms2_rows:
        ms2_annotation = ms2_rows['annotation'].values
    else:
        ms2_annotation = None

    document = []
    for w in range(len(ms2_mz)):
        mz = ms2_mz[w]
        intensity = ms2_intensity[w]
        if not np.isnan(float(ms2_fragment_words[w])):
            fragment_word = 'fragment_' + ms2_fragment_words[w]
            fragment_motif = get_word_motif(fragment_word, ms1_motifs, word_map)
        else:
            fragment_word = np.NaN
            fragment_motif = np.NaN
        if not np.isnan(float(ms2_loss_words[w])):
            loss_word = 'loss_' + ms2_loss_words[w]
            loss_motif = get_word_motif(loss_word, ms1_motifs, word_map)        
        else:
            loss_word = np.NaN
            loss_motif = np.NaN
        if ms2_annotation is not None:
            annot = ms2_annotation[w]
        else:
            annot = np.NaN
        item = (mz, intensity, fragment_word, fragment_motif, loss_word, loss_motif, annot)
        # print "%08.4f   %.2f    %-20s %-5s %-15s %-5s" % item
        document.append(item)

    df = pd.DataFrame(document, columns=['ms2_mz', 'ms2_intensity', 'fragment_word', 'fragment_motif', 'loss_word', 'loss_motif', 'ef'])

    if plotting_func is None:
        plot_fragmentation_spectrum(df, first_row, motif_colour, motif_idx, ms1_label, save_to=save_to, xlim_upper=xlim_upper)
    else:
        plotting_func(df, first_row, motif_colour, motif_idx, ms1_label, save_to=save_to, xlim_upper=xlim_upper, contributions=contributions)
        
    return df
    
def get_peak_ids_of_m2m(G, m2m):
                             
    for node_id, node_data in G.nodes(data=True):

        if node_data['group'] == 2: # is a motif

            motif_name = G.node[node_id]['name']
            _, motif_id = motif_name.split('_')
            motif_id = int(motif_id)
        
            if motif_id == m2m:
                children = G.neighbors(node_id) # get the children of this motif
                children_pids = [int(G.node[c]['peakid']) for c in children]
                return set(children_pids)
                                     
    return None        

def map_standard_to_ms1(std_file, mass_tol, rt_tol, ms2lda, mode='POS', verbose=False):

    std_peaks = []
    with open(std_file, "rb") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        for row in reader:
            stdmix = int(row[0])
            compound_name = row[1]
            formula = row[2]
            polarity = row[3]
            mz = float(row[4])
            rt = float(row[5]) * 60
            mh_intensity = float(row[6])
            tup = (stdmix, compound_name, formula, polarity, mz, rt, mh_intensity)
            if mode == 'POS' and polarity == '+':
                std_peaks.append(tup)
            elif mode == 'NEG' and polarity == '-':
                std_peaks.append(tup)
    
    if verbose:
        for tup in std_peaks:
            print tup

    std = np.array(std_peaks)
    std_mz = np.array([x[4] for x in std_peaks])
    std_rt = np.array([x[5] for x in std_peaks])
    matches = {}
    
    ms1_label = {}
    for row in ms2lda.ms1.itertuples(index=True):
        peakid = row[1]
        mz = row[5]
        rt = row[4]
    
        # the following line is hacky for pos mode data
        mass_delta = mz*mass_tol*1e-6
        mass_start = mz-mass_delta
        mass_end = mz+mass_delta
        rt_start = rt-rt_tol
        rt_end = rt+rt_tol
    
        match_mass = (std_mz>mass_start) & (std_mz<mass_end)
        match_rt = (std_rt>rt_start) & (std_rt<rt_end)
        match = match_mass & match_rt
    
        res = std[match]
        if len(res) == 1:
            closest = tuple(res[0])
            matches[closest] = row
            ms1_label[row[1]] = closest[1]        
        elif len(res)>1:
            closest = None
            min_dist = sys.maxint
            for match_res in res:
                match_mz = float(match_res[4])
                match_rt = float(match_res[5])
                dist = math.sqrt((match_rt-rt)**2 + (match_mz-mz)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest = match_res
            closest = tuple(closest)
            matches[closest] = row
            ms1_label[row[1]] = closest[1]
    
    if verbose:
        print "Matches found %d/%d" % (len(matches), len(std))
        print
    
    ms1_list = []
    for match in matches:
        key = str(match)
        ms1_row = matches[match]
        value = str(ms1_row)
        pid = ms1_row[1]
        if verbose:
            print "Standard %s" % key
            print "MS1 %s" % value
            print
        ms1_list.append(pid)
        
    return ms1_label, ms1_list

def load_motif_annotation(annot_file, verbose=False):

    motif_annotation = {}
    motif_idx = {}
    i = 0
    for item in csv.reader(open(annot_file), skipinitialspace=True):
        key = int(item[0])
        val = item[1]
        if verbose:
            print str(key) + "\t" + val
        motif_annotation[key] = val
        motif_idx[key] = i
        i += 1    
    return motif_annotation, motif_idx

def main():
    
    infile = '/home/joewandy/git/metabolomics_tools/justin/notebooks/results/beer3_pos_rel/beer3_pos_rel_docs.csv'
    # infile = '/home/joewandy/git/metabolomics_tools/justin/notebooks/results/urine37_pos_rel/urine37_pos_rel_docs.csv'
    
    # sif_out = '/home/joewandy/git/metabolomics_tools/justin/cytoscape/beer3_pos_docs_cytoscape.sif'
    # noa_out = '/home/joewandy/git/metabolomics_tools/justin/cytoscape/beer3_pos_docs_cytoscape.noa'
    # export_docdf_to_cytoscape(infile, sif_out, noa_out)
    
    # nodes_out = '/home/joewandy/git/metabolomics_tools/justin/visualisation/beer3_pos_nodes.csv'
    # edges_out = '/home/joewandy/git/metabolomics_tools/justin/visualisation/beer3_pos_edges.csv'
    # export_docdf_to_gephi(infile, nodes_out, edges_out)

    export_docdf_to_networkx(infile)
        
if __name__ == "__main__":
    main()
