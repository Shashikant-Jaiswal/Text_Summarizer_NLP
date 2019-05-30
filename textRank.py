# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 00:11:01 2018

@author: Tejaswini
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from rougescore import rougescore
import glob
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.cluster.util import cosine_distance
from operator import itemgetter


path_train = 'C:\\SHASHI_DATA\\3_NLP\\Coding\\Final_Code\\simulation\\train'
path_test = 'C:\\SHASHI_DATA\\3_NLP\\Coding\\Final_Code\\simulation\\test'

#to generates summary by building similarity matrix 
#and computes text rank to score sentences
#extracts top n sentences as summary
def generate_summary(sentences,vector, index, n=3, stopwords=None):

    """
    sentences =  list of sentences 
    n = number of sentences the summary should contain
    stopwords = list of stopwords
    """
    
    #create similarity matrix
    s = create_similarity_matrix(sentences, vector, index, stop_words) 
    #print(s)
    #compute text rank using similarity matrix
    sentence_ranks = textrank(s)
     
    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    # extracts top n sentences 
    selected_sentences = sorted(ranked_sentence_indexes[:n])
    #print(selected_sentences)
    #build summary from selected top n sentences
    summary = itemgetter(*selected_sentences)(sentences)
    #print(summary)
    return summary

#to compute text rank using similarity matrix
#d: damping factor
#S: Similarity matrix
def textrank(S, d=0.85):
    R = np.ones(len(S)) / len(S)

    #iterating over 10 times to converge text rank value
    for i in range(1,10):
        new_R = np.ones(len(S)) * (1 - d) / len(S) + d * S.T.dot(R)
        R = new_R
    return R

#compute sentence similarity using cosine similarity
#cosine similarity = 1- cosine distance
def compute_sentence_similarity(sent1, sent2,vector,text_index,stopwords=None):
    
    # Get feature names from the training TFIDF vector
    feature_names = vectorizer.get_feature_names()
    doc = 0
    # Get the feature index for the current test document
    feature_index = vector.nonzero()[1]
    # Get TFIDF scores for the words in current test document
    tfidf_scores = zip(feature_index, [vector[doc, x] for x in feature_index])
    
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        flag = False
        if w in stopwords:
            continue
        # Iterate over the words in the TFIDF vector of the current document
        for word, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            # check if the current word and word in TFIDF vectors are same
            if(word==w):
                val = s
                flag = True
                break
        # If we found the word in TFIDF matrix assign it to the vector
        if(flag==True):
             vector1[all_words.index(w)] += val
        else:
             vector1[all_words.index(w)] += 1

            
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        # Iterate over the words in the TFIDF vector of the current document
        for word, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            if(word==w):
                val = s
                flag = True
                break
        # If we found the word in TFIDF matrix assign it to the vector
        if(flag==True):
             vector2[all_words.index(w)] += val
        else:
             vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


# create similarity matrix using sentence similarity between each pair of sentences in text    
def create_similarity_matrix(sentences,vector,index, stopwords=None):
    # Initialize similarity matrix with value = 0
    # size of similarity matrix : nXn
    # n: length of sentence
    s = np.zeros((len(sentences), len(sentences)))
    # compute values of s[i][j] to create similarity matrix
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            
            #S[idx1][idx2] = compute_cs(sentences[idx1], sentences[idx2])
            s[i][j] = compute_sentence_similarity(sentences[i], sentences[j],vector, index, stop_words)
 
    # normalize the matrix row-wise
    for i in range(len(s)):
        s[i] /= s[i].sum()
 
    return s


#Retreive news articles from training data set 

corpus_train=[]
i=0
for filename in sorted(glob.glob(os.path.join(path_train, '*.sent'))):
    file=open(filename,"r",encoding="utf8")
    text=file.read()
    #print(text)
    print(i)
    #corpus[filename]=text
    corpus_train.append(text)
    i+=1

#Retreive news articles from test data set

corpus_test=[]
i=0
for filename in sorted(glob.glob(os.path.join(path_test, '*.sent'))):
    file=open(filename,"r",encoding="utf8")
    text=file.read()
    #print(text)
    print(i)
    #corpus[filename]=text
    corpus_test.append(text)
    i+=1


#Retreive summary of news articles from test data set    
corpus_test_summary=[]
for filename in sorted(glob.glob(os.path.join(path_test, '*.summ'))):
    file=open(filename,"r",encoding="utf8")
    text=file.read()
    #print(text)
    print(i)
    corpus_test_summary.append(text)
    i+=1 

stop_words = stopwords.words('english')
#Initialize vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
# tokenize and build vocab
vectorizer.fit(corpus_train)
#print(vectorizer.vocabulary)
vector=[]
sentences=[]
summary=[]
reference=[]
#Initialize total precision and total recall values
precision_total=0
recall_total=0
rscore_total=0.0
#Itereate over list of news articles in test corpus and generate corresponding summary for each article
for index in range(len(corpus_test)):
    # transform document into vector
    vector= vectorizer.transform([corpus_test[index]]).toarray()
    #tokenize text into senetences
    sentences=[sent for sent in sent_tokenize(corpus_test[index])]
    #print(sentences)
    #print(len(sentences))
    summ=""
    #generates summary by computing sentence scores using text rank
    for i, sentence in enumerate(generate_summary(sentences,vector,index,stopwords=stopwords.words('english'))):
        summ+=sentence
    #summary.append(summ)
    #The system generated summary
    predicted_summary = summ
    # retreive the actual summary of corresponding news article
    actual_summary = corpus_test_summary[index]
    #print("The Reference summary:")
    #print(actual_summary)
    #print("The model generated summary:")
    #print(predicted_summary)
    #compute Rouge score
    rscore = rougescore.rouge_n(predicted_summary, actual_summary, 1, 0.0)
    rscore_total += rscore

    
avg_rscore = rscore_total / len(corpus_test)

print("\n\n")
#print("corpus length :" + str(len(corpus_test)))
#print("total Rscore is :" + str(rscore_total))
print("Rouge score for summarization is:" + str(avg_rscore))