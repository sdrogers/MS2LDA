"""
Visualisation methods for the LDA results
"""
from __future__ import print_function

from pandas.core.frame import DataFrame
import sys

import numpy as np
import pandas as pd
import pylab as plt


def export_docdf(infile, sif_out, noa_out):
    """ Exports docdf to a format that can be parsed by cytoscape """

    print("Loading " + infile)
    docdf = pd.read_csv(infile, index_col=0)
    
    df = DataFrame(columns=('parent', 'child', 'score'))    
    i = 0
    docnames = []
    topics = []
    for row_index, row_value in docdf.iterrows():    
        for col_index, col_value in row_value.iteritems():
            if col_value > 0:
                tokens = row_index.split('_')
                docname = str(tokens[0]) + "_" + str(tokens[1])
                topic = col_index
                score = col_value 
                df.loc[i] = [topic, docname, score]
                docnames.append(docname)
                topics.append(topic)
                i += 1
#     unique = set(topics)
#     for u in unique:
#         df.loc[i] = ["ROOT", u, 0]
#         i += 1
                
    print(df)    
    df.to_csv(sif_out, index=False)
    print("Saved to " + sif_out)
    
    f = open(noa_out,'w')
    print("NodeType", file=f)
    for docname in docnames:
        print(docname + " = document", file=f)
    for topic in topics:
        print(topic + " = topic", file=f)
    f.close()    
    print("Saved to " + noa_out)
    
def main():
    
    infile = '/home/joewandy/git/metabolomics_tools/justin/results/beer3_pos/beer3_pos_docs.csv'
    sif_out = '/home/joewandy/git/metabolomics_tools/justin/cytoscape/beer3_pos_docs_cytoscape.sif'
    noa_out = '/home/joewandy/git/metabolomics_tools/justin/cytoscape/beer3_pos_docs_cytoscape.noa'
    export_docdf(infile, sif_out, noa_out)

        
if __name__ == "__main__":
    main()