# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 19:28:38 2021

@author: Fritz
"""
import timeit

class Algorithms:
    def LogisticRegression(finishedText,dataset_Var):
                
        tic=timeit.default_timer()
                
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0)
        classifier.fit(finishedText[0], finishedText[3]) 
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        text_classifier = Pipeline([('tfidf',TfidfVectorizer()),('classifier',LogisticRegression())])
        
        text_classifier.fit(finishedText[1], finishedText[3])
        
        classElem = text_classifier.predict(dataset_Var)
        classElemOri = finishedText[5].inverse_transform(classElem)
        classProb = text_classifier.predict_proba(dataset_Var)
                
        classProb = [max(p) for p in classProb] 
        classProb = [round(num, 3) for num in classProb]
        
        y_pred = text_classifier.predict(finishedText[2])

        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        cm = confusion_matrix(finishedText[4], y_pred)
        acc = accuracy_score(finishedText[4], y_pred)
        pre = precision_score(finishedText[4], y_pred, average='macro')
        rec = recall_score(finishedText[4], y_pred, average='macro')
        f1 = f1_score(finishedText[4], y_pred, average='macro')
        print(cm)

        toc=timeit.default_timer()
        runtime = toc - tic
                
        classElemOri = str(classElemOri)
        classProb = str(classProb)
                
        return acc, runtime,classElemOri, classProb, pre, rec,f1
            
    def kNN(finishedText,dataset_Var,TestValue):
            
        tic=timeit.default_timer()
                    
        from sklearn.neighbors import KNeighborsClassifier
        
        neighbors_numbers = TestValue/20
        
        classifier = KNeighborsClassifier(n_neighbors = int(neighbors_numbers), metric = 'minkowski', p = 2)
        classifier.fit(finishedText[0], finishedText[3])
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        text_classifier = Pipeline([('tfidf',TfidfVectorizer()),('classifier',KNeighborsClassifier())])
        
        text_classifier.fit(finishedText[1], finishedText[3])
        
        classElem = text_classifier.predict(dataset_Var)
        classElemOri = finishedText[5].inverse_transform(classElem)
        classProb = text_classifier.predict_proba(dataset_Var)
                
        classProb = [max(p) for p in classProb] 
        classProb = [round(num, 3) for num in classProb]

        y_pred = text_classifier.predict(finishedText[2])

        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        cm = confusion_matrix(finishedText[4], y_pred)
        acc = accuracy_score(finishedText[4], y_pred)
        pre = precision_score(finishedText[4], y_pred, average='macro')
        rec = recall_score(finishedText[4], y_pred, average='macro')
        f1 = f1_score(finishedText[4], y_pred, average='macro')   
        print(cm)
        
        toc=timeit.default_timer()
        runtime = toc - tic
            
        classElemOri = str(classElemOri)
        classProb = str(classProb)
            
        return acc, runtime, classElemOri, classProb, pre, rec, f1
        
    def SVM(finishedText,dataset_Var):
            
        tic=timeit.default_timer()
            
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(finishedText[0], finishedText[3])
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        text_classifier = Pipeline([('tfidf',TfidfVectorizer()),('classifier',SVC(probability=True))])
        
        text_classifier.fit(finishedText[1], finishedText[3])
            
        classElem = text_classifier.predict(dataset_Var)
        classElemOri = finishedText[5].inverse_transform(classElem)
        classProb = text_classifier.predict_proba(dataset_Var)
            
        classProb = [max(p) for p in classProb] 
        classProb = [round(num, 3) for num in classProb]
            
        y_pred = text_classifier.predict(finishedText[2])

        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        cm = confusion_matrix(finishedText[4], y_pred)
        acc = accuracy_score(finishedText[4], y_pred)
        pre = precision_score(finishedText[4], y_pred, average='macro')
        rec = recall_score(finishedText[4], y_pred, average='macro')
        f1 = f1_score(finishedText[4], y_pred, average='macro')      
        print(cm)
            
        toc=timeit.default_timer()
        runtime = toc - tic

        classElemOri = str(classElemOri)
        classProb = str(classProb)
            
        return acc, runtime, classElemOri, classProb, pre, rec, f1
    
    def NaiveBayes(finishedText,dataset_Var):
            
        from sklearn.preprocessing import binarize
        
        tic=timeit.default_timer()
            
        from sklearn.naive_bayes import BernoulliNB
        classifier = BernoulliNB()
        
        print(type(finishedText[0]))

        training_X = binarize(finishedText[0])

        classifier.fit(training_X, finishedText[3])
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        text_classifier = Pipeline([('tfidf',TfidfVectorizer()),('classifier',BernoulliNB())])
        
        text_classifier.fit(finishedText[1], finishedText[3])
        
        classElem = text_classifier.predict(dataset_Var)
        classElemOri = finishedText[5].inverse_transform(classElem)
        classProb = text_classifier.predict_proba(dataset_Var)
            
        classProb = [max(p) for p in classProb] 
        classProb = [round(num, 3) for num in classProb]

        y_pred = text_classifier.predict(finishedText[2])

        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        cm = confusion_matrix(finishedText[4], y_pred)
        acc = accuracy_score(finishedText[4], y_pred)
        pre = precision_score(finishedText[4], y_pred, average='macro')
        rec = recall_score(finishedText[4], y_pred, average='macro')
        f1 = f1_score(finishedText[4], y_pred, average='macro')        
        print(cm)
            
        toc=timeit.default_timer()
        runtime = toc - tic
        
        classElemOri = str(classElemOri)
        classProb = str(classProb)

        return acc, runtime, classElemOri, classProb, pre, rec, f1
    
    def FNN(datasetANN,predElem,dataset_Var):
        
        tic=timeit.default_timer()
        import tensorflow as tf
        tf.__version__
        import pandas as pd
        import numpy as np
        import tensorflow_hub as hub
        
        from sklearn.model_selection import train_test_split
        
        dataset = datasetANN[["ProdDescModelName",predElem]]
        dataset = dataset.dropna()
        print(type(dataset))
        print(dataset.columns)
        pd.set_option('display.max_colwidth', -1)
        print(dataset)
        dataset_lenght = len(dataset)
        
        X_train, X_test = train_test_split(dataset, test_size=0.2,random_state=111)
        
        from sklearn.utils import class_weight
        
        class_weights = list(class_weight.compute_class_weight('balanced',np.unique(dataset[predElem]),dataset[predElem]))
        
        print(dataset[predElem].value_counts())
        class_weights.sort()
        print(np.unique(dataset[predElem].tolist()))
        print(class_weights)
        
        weights={}
        
        for index, weight in enumerate(class_weights):
            weights[index]=weight
            
        print(weights)
        
        dataset_train = tf.data.Dataset.from_tensor_slices((X_train['ProdDescModelName'].tolist(), X_train[predElem].tolist()))
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test['ProdDescModelName'].tolist(), X_test[predElem].tolist()))
        
        for text,target in dataset_train.take(5):
            print('Complaint: {}, Target: {}'.format(text, target))
            
        for text,target in dataset_test.take(5):
            print('Complaint: {}, Target: {}'.format(text, target))
            
        numbers = range(0, len(np.unique(dataset[predElem].tolist())))
        
        print(numbers)
    
        sequence_of_numbers = []
        for number in numbers:
            sequence_of_numbers.append(number)
            
        print(sequence_of_numbers)
               
            
        table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(np.unique(dataset[predElem].tolist())),
                values=tf.constant(sequence_of_numbers),
            ),
            default_value=tf.constant(-1),
            name="target_encoding"
        )
        
        lenght_sequence_of_numbers = len(sequence_of_numbers)
        
        @tf.function
        def target(x):
          return table.lookup(x)
        
        def show_batch(dataset, size=5):
          for batch, label in dataset.take(size):
              print(batch.numpy())
              print(target(label).numpy())
              
        show_batch(dataset_test,6)
        
        def fetch(text, labels):
            return text, tf.one_hot(target(labels),lenght_sequence_of_numbers)
        
        train_data_f=dataset_train.map(fetch)
        test_data_f=dataset_test.map(fetch)
        
        print(next(iter(train_data_f)))
        
        train_data, train_labels = next(iter(train_data_f.batch((5))))
        print(train_data, train_labels)

        print(lenght_sequence_of_numbers)   
        
        embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
        hub_layer = hub.KerasLayer(embedding, output_shape=[128], input_shape=[], 
                                   dtype=tf.string, trainable=True)
        hub_layer(train_data[:1])
                
        model = tf.keras.Sequential()
        model.add(hub_layer)
        for units in [128, 128, 64 , 32]:
          model.add(tf.keras.layers.Dense(units, activation='relu'))
          model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(lenght_sequence_of_numbers, activation='softmax'))
        
        model.summary()
        
        model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['acc'])
        
        train_data_f=train_data_f.shuffle(dataset_lenght).batch(512)
        test_data_f=test_data_f.batch(512)
        
        history = model.fit(train_data_f,
                            epochs=100,
                            validation_data=test_data_f,
                            verbose=1,
                            class_weight=weights)
        
        print(len(list(dataset_test)))
                
        Xnew = dataset_Var
        classElem = model.predict_classes(Xnew)
        classElem = np.array_str(classElem)
        classElem = classElem.replace("[","") 
        classElem = classElem.replace("]","") 
        classElem = int(classElem)
        print(classElem)
        print(type(classElem))
        
        zip_iterator = zip(sequence_of_numbers, np.unique(dataset[predElem].tolist()))      
        a_dictionary = dict(zip_iterator)
        print(a_dictionary)
        classElem = a_dictionary[classElem]
        
        ynew_proba = model.predict_proba(Xnew)
        classProb = [max(p) for p in ynew_proba] 
        classProb = [round(num, 3) for num in classProb]
        
        results = model.evaluate(dataset_test.map(fetch).batch(dataset_lenght), verbose=2)

        print(results)
        
        test_data, test_labels = next(iter(dataset_test.map(fetch).batch(dataset_lenght)))
        
        y_pred=model.predict(test_data)
        from sklearn.metrics import classification_report
        print(classification_report(test_labels.numpy().argmax(axis=1), y_pred.argmax(axis=1)))
        
        from sklearn.metrics import precision_recall_fscore_support as score
        from sklearn.metrics import accuracy_score
        
        accuracy = accuracy_score(test_labels.numpy().argmax(axis=1),y_pred.argmax(axis=1))
        print(accuracy)

        precision,recall,f1,support=score(test_labels.numpy().argmax(axis=1),y_pred.argmax(axis=1),average='macro')
        print('Precision : {}'.format(precision))
        print( 'Recall    : {}'.format(recall))
        print( 'F-score   : {}'.format(f1))
        print( 'Support   : {}'.format(support))
        
        
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(test_labels.numpy().argmax(axis=1), y_pred.argmax(axis=1)))

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        
        def plot_history(history,predElem):
            acc = history.history['acc']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            x = range(1, len(acc) + 1)
        
            plt.figure(figsize=(12, 5))
            plt.suptitle(str(len(dataset))+" data - "+str(Algorithms.FNN.__name__)+" - "+str(predElem), fontsize=20)
            plt.subplot(1, 2, 1)
            plt.plot(x, acc, 'b', label='Training acc')
            plt.plot(x, val_acc, 'r', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(x, loss, 'b', label='Training loss')
            plt.plot(x, val_loss, 'r', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()
        
        plot_history(history,predElem)
                   
        toc=timeit.default_timer()
        runtime = toc - tic
        
        classElem = str(classElem)
        classProb = str(classProb)
        
       
        return accuracy, runtime, precision, recall, f1, classElem, classProb
