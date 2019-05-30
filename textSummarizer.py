# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:32:53 2018

@author: Tejaswini Nardella, Anusha Balaji, Shashikant Jaiswal
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from collections import OrderedDict
#from rouge import Rouge
from rougescore import rougescore
import glob
import os

#Retreive news articles from training data set

path_train = 'C:\\SHASHI_DATA\\3_NLP\\Coding\\Final_Code\\data\\train'
path_test = 'C:\\SHASHI_DATA\\3_NLP\\Coding\\Final_Code\\data\\test'

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

#Initialize vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
# tokenize and build vocab from training corpus
vectorizer.fit(corpus_train)
#print(vectorizer.vocabulary)
stop_words=set(stopwords.words('english'))
#Initialize total rscore value
rscore_total=0.0
#Itereate over list of news articles in test corpus and generate corresponding summary for each article
for i in range(len(corpus_test)):
    # transform document into vector
    vector= vectorizer.transform([corpus_test[i]]).toarray()
    # Sort the weights from highest to lowest: sorted_tfidf_weights
    #sorted_tfidf_weights = sorted(vector, key=lambda w: w[1], reverse=True)
    #build sentence score dictionary
    sentScore=dict()    
    #Iterate over all sentences to compute sentence score of each sentence in text
    for sent in sent_tokenize(corpus_test[i]):
        for w in word_tokenize(sent):
            if w not in stop_words:
                index=vectorizer.vocabulary_.get(w)
                if(index!=None):
                    w_score=vector[0][index]
                    print(index)
                    if sent[0:15] in sentScore:
                        sentScore[sent]+=w_score
                    else:
                        sentScore[sent]=w_score
    #sort the items in dictionary based on sentence score    
    sorted_dict = OrderedDict(sorted(sentScore.items(), key=lambda x: x[1],reverse=True))

    #print("\n\nSummary:\n")
    count=1
    summ=""
    
    #retreive top 5 highest score sentences and generate summary
    for k, v in sorted_dict.items():
        if(count>3):
            break  
        #print("%s. %s" % ((i), ''.join(k)))
        summ+=k
        count+=1
    #print("The system generated summary:")
    #print(summ)
    predicted_summary=summ
    #retreive the actual summary of corresponding news article
    actual_summary=corpus_test_summary[i]
    #print("The Reference summary:")
    #print(actual_summary)
    #print("The model generated summary:")
    #print(predicted_summary)

    #compute precision and recall
    rscore= rougescore.rouge_n(predicted_summary,actual_summary,1,0.0)
    rscore_total+=rscore

avg_rscore= rscore_total/len(corpus_test)

print("\n\n")
print("Test corpus length :"+str(len(corpus_test)))
#print("total Rscore is :"+str(rscore_total))
print("Rouge score for summarization is :"+str(avg_rscore))
    
    

    

