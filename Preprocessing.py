# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:33:10 2021

@author: Fritz
"""

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import timeit

class PreProcessing:   
    def PreProcessingANN(TestValue):  
        
        file = open("PTR-200k.csv", encoding="utf8")
        numline = len(file.readlines())
        print (numline)
            
        nlinesfile = numline
        nlinesrandomsample = TestValue
        lines2skip = np.random.choice(np.arange(1,nlinesfile+1), (nlinesfile-nlinesrandomsample), replace=False)
            
        dataset = pd.read_csv("PTR-200k.csv"
                              ,na_values=" NaN"
                              ,nrows=TestValue
                              ,skiprows=lines2skip
                              ,encoding="utf8"
                              ,error_bad_lines=False)
        
        dataset['ProdDescModelName'] = dataset['Product Description'] + ' ' + dataset['Model Name']
        
        return dataset
    
    def PreProcessingML(dataset):  
        
        tic=timeit.default_timer()

        porter = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        
        R = dataset["ProdDescModelName"] 
            
        R = R.dropna()
        
        print(R)
        
        from spacy.lang.en.stop_words import STOP_WORDS
        
        def stemSentence(line):
                    token_words= word_tokenize(line)
                    token_words = [word for word in token_words if not word in STOP_WORDS]
                    token_words
                    stem_sentence=[]
                    for word in token_words:
                        lemmatizer.lemmatize(word)
                        stem_sentence.append(porter.stem(word))
                        stem_sentence.append(" ")
                    return "".join(stem_sentence)   
                
        dataset["ProdDescModelName"]= R.apply(stemSentence)
        
        toc=timeit.default_timer()
        runtime = toc - tic
        runtime = round(runtime, 3)
        runtime = str(runtime)
            
        finished = "Finished Processing Data in " + runtime + " seconds"
            
        return dataset, finished
            
    def FeatureExtraction(dataset, predElem):

            print(dataset)
            
            R = dataset[['ProdDescModelName',predElem]] 
            
            R = R.dropna()

            X = R['ProdDescModelName']
            print(X)
                                   
            y = R[predElem]
            
            print(y)

            from sklearn.preprocessing import LabelEncoder
            
            le = LabelEncoder()
            y = le.fit_transform(y)            
            
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
            
            from sklearn.feature_extraction.text import CountVectorizer
            
            count_vect = CountVectorizer()
            X_train_counts = count_vect.fit_transform(X_train)
            X_train_counts.shape
    
            from sklearn.feature_extraction.text import TfidfTransformer
            tfidf_transformer = TfidfTransformer()
            X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
            X_train_tfidf.shape
            
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer()
            X_train_tfidf = vectorizer.fit_transform(X_train)
            
            print("Training/Test Data:")
            print (X_train_tfidf)
            print(X_test)
            print("")
            print(y_train)
            print(y_test)
            print("")
            
            return X_train_tfidf,X_train, X_test, y_train, y_test, le
        
        