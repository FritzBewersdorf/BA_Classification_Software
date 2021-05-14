# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 00:37:05 2021

@author: Fritz
"""

# Use Tkinter for python 2, tkinter for python 3
from tkinter import Tk, Label, Button, Entry
from tkinter import ttk
from Preprocessing import *
from Algorithms import *
import csv
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

class GUI:
    def __init__(self, master, *args, **kwargs):
        self.master = master
                                    
        self.TextInsert_Label = Label(master, text="Insert text ")                       
        self.Model_Label_1 = Label(master, text="Logistic Regression")
        self.Model_Label_2 = Label(master, text="kNN")
        self.Model_Label_3 = Label(master, text="SVM")
        self.Model_Label_4 = Label(master, text="Naive Bayes")
        self.Model_Label_5 = Label(master, text="Feedforward Neural Network")
                            
        self.predElement_1_1_1 = Label(master)
        self.predElement_1_1_2 = Label(master)
        self.predElement_1_2_1 = Label(master)
        self.predElement_1_2_2 = Label(master)
        self.predElement_1_3_1 = Label(master)
        self.predElement_1_3_2 = Label(master)
        self.predElement_1_4_1 = Label(master)
        self.predElement_1_4_2 = Label(master)
        self.predElement_1_5_1 = Label(master)
        self.predElement_1_5_2 = Label(master)
        self.predElement_1_6_1 = Label(master)
        self.predElement_1_6_2 = Label(master)
                            
        self.dataAmount = ttk.Combobox(master, values=(10, 100, 1000, 5000,10000,50000,
                                                100000,200000),width=22)
                                
        self.Wiedergabe = Label(master)
        self.Wiedergabefinished = Label(master)
                    
        self.inputfield = Entry(master, bd=5, width=40)
                
        self.classify_button = Button(master, text="Classify",
            command=lambda: GUI.initialization(self, master, self.inputfield,
                                                           self.Wiedergabe,self.dataAmount,
                                                           self.Wiedergabefinished))
                                                        
        self.TextInsert_Label.grid(row = 0, column = 0)
        self.Model_Label_1.grid(row = 5, column = 1)
        self.Model_Label_2.grid(row = 5, column = 2)
        self.Model_Label_3.grid(row = 5, column = 3)
        self.Model_Label_4.grid(row = 5, column = 4)
        self.Model_Label_5.grid(row = 5, column = 5)
        self.predElement_1_1_1.grid(row = 3, column = 1)
        self.predElement_1_1_2.grid(row = 4, column = 1)
        self.predElement_1_2_1.grid(row = 3, column = 2)
        self.predElement_1_2_2.grid(row = 4, column = 2)
        self.predElement_1_3_1.grid(row = 3, column = 3)
        self.predElement_1_3_2.grid(row = 4, column = 3)
        self.predElement_1_4_1.grid(row = 3, column = 4)
        self.predElement_1_4_2.grid(row = 4, column = 4)
        self.predElement_1_5_1.grid(row = 3, column = 5)
        self.predElement_1_5_2.grid(row = 4, column = 5)
        self.inputfield.grid(row = 0, column = 1)
        self.dataAmount.grid(column=3, row=0)
        self.dataAmount.current(1)
        self.classify_button.grid(row = 1, column = 1)
        self.Wiedergabe.grid(row = 2, column = 1)
        self.Wiedergabefinished.grid(row = 2, column = 3)
            
    def initialization(self, master, inputfield,Wiedergabe,dataAmount,Wiedergabefinished):
        try:
            entry_text = inputfield.get()
            if (entry_text == ""):
                Wiedergabe.config(text="Insert Text!", fg="red")
            else:           
                
                with open('New_Input.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["input_text"])
                    writer.writerow([entry_text])
                    
                dataset_Var = pd.read_csv("New_Input.csv",na_values=" NaN")
                
                from spacy.lang.en.stop_words import STOP_WORDS
                
                
                porter = PorterStemmer()
                lemmatizer = WordNetLemmatizer()
                
                def NLPInputText(line):
                    token_words= nltk.word_tokenize(line)
                    token_words = [word for word in token_words if not word in STOP_WORDS]
                    token_words
                    stem_sentence=[]
                    for word in token_words:
                        lemmatizer.lemmatize(word)
                        stem_sentence.append(porter.stem(word))
                        stem_sentence.append(" ")
                    return "".join(stem_sentence)   
                
                dataset_Var["input_text"]= dataset_Var["input_text"].apply(NLPInputText)

                predElem = ["Product Division","Product Group","Sport","Product Type","Gender","Age Group","Main Color"]

                TestValue = int(dataAmount.get())
                print(TestValue) 
                print(dataset_Var["input_text"])
                
                datasetANN = PreProcessing.PreProcessingANN(TestValue)
                dataset = PreProcessing.PreProcessingML(datasetANN)

                Wiedergabefinished.config(text=dataset[1] ,fg="green")
                
                for iteration,predElem in enumerate(predElem):
                              
                    finishedText = PreProcessing.FeatureExtraction(dataset[0],predElem)
                      
                    accLogReg = Algorithms.LogisticRegression(finishedText,dataset_Var)
                    runtime = round(accLogReg[1],3)
                    runtimeall = runtime + runtimeall
                    acc= round(accLogReg[0],3)             
                    accall = acc + accall
                    pre= round(accLogReg[4],3)             
                    preall = pre + preall
                    rec= round(accLogReg[5],3)             
                    recall = rec + recall
                    f1= round(accLogReg[6],3)             
                    f1all = f1 + f1all
                    accLogRegValue = accLogReg[2]," Confidence: "+accLogReg[3]
                    
                    acckNN = Algorithms.kNN(finishedText,dataset_Var,TestValue)
                    runtimekNN = round(acckNN[1],3)
                    runtimeallkNN = runtimekNN + runtimeallkNN
                    acckNN1 = round(acckNN[0],3)             
                    accallkNN = acckNN1 + accallkNN
                    prekNN= round(acckNN[4],3)             
                    preallkNN = prekNN + preallkNN
                    reckNN= round(acckNN[5],3)             
                    recallkNN = reckNN + recallkNN
                    f1kNN= round(acckNN[6],3)             
                    f1allkNN = f1kNN + f1allkNN
                    acckNNValue = acckNN[2], " Confidence: "+acckNN[3]
                    
                    accSVM = Algorithms.SVM(finishedText,dataset_Var)
                    runtimeSVM = round(accSVM[1],3)
                    runtimeallSVM = runtimeSVM + runtimeallSVM
                    accSVM1 = round(accSVM[0],3)             
                    accallSVM = accSVM1 + accallSVM
                    preSVM= round(accSVM[4],3)             
                    preallSVM = preSVM + preallSVM
                    recSVM= round(accSVM[5],3)             
                    recallSVM = recSVM + recallSVM
                    f1SVM = round(accSVM[6],3)             
                    f1allSVM = f1SVM + f1allSVM
                    accSVMValue = accSVM[2], " Confidence: "+ accSVM[3]
                    
                    accNaive_Baines = Algorithms.NaiveBayes(finishedText,dataset_Var)
                    runtimeBaines = round(accNaive_Baines[1],3)
                    runtimeallBaines = runtimeBaines + runtimeallBaines
                    accBaines = round(accNaive_Baines[0],3)             
                    accallBaines = accBaines + accallBaines
                    preBaines= round(accNaive_Baines[4],3)             
                    preallBaines = preBaines + preallBaines
                    recBaines= round(accNaive_Baines[5],3)             
                    recallBaines = recBaines + recallBaines
                    f1Baines = round(accNaive_Baines[6],3)             
                    f1allBaines = f1Baines + f1allBaines
                    accNaive_BainesValue = accNaive_Baines[2], " Confidence: "+ accNaive_Baines[3]
                    
                    accFNN = Algorithms.FNN(datasetANN,predElem,dataset_Var)
                    runtimeFNN = round(accFNN[1],3)
                    runtimeallFNN = runtimeFNN + runtimeallFNN
                    accFNN1 = round(accFNN[0],3)             
                    accallFNN = accFNN1 + accallFNN
                    preFNN= round(accFNN[2],3)             
                    preallFNN = preFNN + preallFNN
                    recFNN= round(accFNN[3],3)             
                    recallFNN = recFNN + recallFNN
                    f1FNN = round(accFNN[4],3)             
                    f1allFNN = f1FNN + f1allFNN
                    accFNNValue = accFNN[5], " Confidence: "+ accFNN[6]   
                    
                    self.ValueElement_0 = Label(master)
                    self.ValueElement_0.grid(row=iteration+6, column=0)
                    self.ValueElement_0.config(text=predElem)
                      
                    self.ValueElement_1 = Label(master)
                    self.ValueElement_1.grid(row=iteration+6, column=1)
                    self.ValueElement_1.config(text=accLogRegValue)
                    
                    self.ValueElement_2 = Label(master)
                    self.ValueElement_2.grid(row=iteration+6, column=2)
                    self.ValueElement_2.config(text=acckNNValue)
                    
                    self.ValueElement_3 = Label(master)
                    self.ValueElement_3.grid(row=iteration+6, column=3)
                    self.ValueElement_3.config(text=accSVMValue)
                    
                    self.ValueElement_4 = Label(master)
                    self.ValueElement_4.grid(row=iteration+6, column=4)
                    self.ValueElement_4.config(text=accNaive_BainesValue)
                    
                    self.ValueElement_4 = Label(master)
                    self.ValueElement_4.grid(row=iteration+6, column=5)
                    self.ValueElement_4.config(text=accFNNValue)
                    
                iteration = iteration +1
                
                runtimeall = round(runtimeall,3)
                runtimeall = str(runtimeall)
                
                elementlist = [accall, preall,recall,f1all]
                
                for iteration2, element in enumerate(elementlist):
                    element = element/iteration
                    element = round(element,3)
                    elementlist[iteration2] = element

                accLogRegNeu = "Accuracy: "+str(elementlist[0])," Runtime: "+str(runtimeall)
                accLogRegNeu2 = "Precision: "+str(elementlist[1]) , " Recall: "+str(elementlist[2])," F1-Score: "+str(elementlist[3])
                self.predElement_1_1_1.config(text=accLogRegNeu) 
                self.predElement_1_1_2.config(text=accLogRegNeu2)   
                
                runtimeallkNN = round(runtimeallkNN,3)
                runtimeallkNN = str(runtimeallkNN)
                
                elementlistkNN = [accallkNN, preallkNN,recallkNN,f1allkNN]
                
                for iteration3, element in enumerate(elementlistkNN):
                    element = element/iteration
                    element = round(element,3)
                    elementlistkNN[iteration3] = element
                
                acckNNNeu = "Accuracy: "+str(elementlistkNN[0])," Runtime: "+ str(runtimeallkNN)
                acckNNNeu2 = "Precision: "+str(elementlistkNN[1]) , " Recall: "+str(elementlistkNN[2])," F1-Score: "+str(elementlistkNN[3])
                self.predElement_1_2_1.config(text=acckNNNeu)
                self.predElement_1_2_2.config(text=acckNNNeu2)
                
                runtimeallSVM = round(runtimeallSVM,3)
                runtimeallSVM = str(runtimeallSVM)
                
                elementlistSVM = [accallSVM, preallSVM,recallSVM,f1allSVM]
                
                for iteration4, element in enumerate(elementlistSVM):
                    element = element/iteration
                    element = round(element,3)
                    elementlistSVM[iteration4] = element
                
                accSVMNeu = "Accuracy: "+str(elementlistSVM[0])," Runtime: "+ str(runtimeallSVM)
                accSVMNeu2 = "Precision: "+str(elementlistSVM[1]) , " Recall: "+str(elementlistSVM[2])," F1-Score: "+str(elementlistSVM[3])
                self.predElement_1_3_1.config(text=accSVMNeu)
                self.predElement_1_3_2.config(text=accSVMNeu2)
                
                runtimeallBaines = round(runtimeallBaines,3)
                runtimeallBaines = str(runtimeallBaines)
                
                elementlistBaines = [accallBaines, preallBaines,recallBaines,f1allBaines]
                
                for iteration5, element in enumerate(elementlistBaines):
                    element = element/iteration
                    element = round(element,3)
                    elementlistBaines[iteration5] = element
                
                accNaive_BainesNeu = "Accuracy: "+str(elementlistBaines[0])," Runtime: "+ str(runtimeallBaines)
                accNaive_BainesNeu2 = "Precision: "+str(elementlistBaines[1]) , " Recall: "+str(elementlistBaines[2])," F1-Score: "+str(elementlistBaines[3])
                self.predElement_1_4_1.config(text=accNaive_BainesNeu)
                self.predElement_1_4_2.config(text=accNaive_BainesNeu2)
                
                runtimeallFNN = round(runtimeallFNN,3)
                runtimeallFNN = str(runtimeallFNN)
       
                elementlistFNN = [accallFNN, preallFNN,recallFNN,f1allFNN]
                
                for iteration6, element in enumerate(elementlistFNN):
                    element = element/iteration
                    element = round(element,3)
                    elementlistFNN[iteration6] = element
                
                accFNNNeu = "Accuracy: "+str(elementlistFNN[0])," Runtime: "+ str(runtimeallFNN)
                accFNNNeu2 = "Precision: "+str(elementlistFNN[1]) , " Recall: "+str(elementlistFNN[2])," F1-Score: "+str(elementlistFNN[3])
                self.predElement_1_5_1.config(text=accFNNNeu)
                self.predElement_1_5_2.config(text=accFNNNeu2)

                
        except:
           Wiedergabe.config(text="Please open the software again or contact a dev", fg="red")
    
if __name__ == "__main__":
    root = Tk()
    mygui = GUI(root)
    root.title("GUI for NLP classification")
    root.geometry('1800x500')
    root.mainloop()
    