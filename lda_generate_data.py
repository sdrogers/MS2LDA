import sys

import wikipedia

import numpy as np
import pandas as pd
import pylab as plt


class LdaDataGenerator:
    
        def __init__(self, topic_list, document_length, alpha, beta):
            
            self.topic_list = topic_list
            self.document_length = document_length
            self.alpha = alpha
            self.beta = beta
            
            topics = []
            for t in topic_list:
                topic = wikipedia.page(t)
                print topic.title
                print topic.content
                topics.append(topic)
                
def main():

    topic_list = ['London', 'New York']
    gen = LdaDataGenerator(topic_list, 100, 0.1, 0.1)

if __name__ == "__main__":
    main()                