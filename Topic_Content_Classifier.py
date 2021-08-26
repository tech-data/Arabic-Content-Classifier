#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
DF=pd.read_csv('nlpdata.csv')


# In[16]:


DF.isna().sum()


# In[17]:


#delete nan values
DF.dropna(inplace=True) 


# In[18]:


DF.isna().sum()


# In[19]:


#import seaborn as sns
#sns.countplot(x= 'Topic',data = DF)


# In[20]:


import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from nltk.tokenize import regexp_tokenize 

def normalizer(sen):
    sen = re.sub(r'[^\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD ]+', '', sen)# keep only arabic letters and spaces 
    sen=re.sub(r'\b\w{1,3}\b', ' ', sen).strip() #delete words that less than 4 letter
    tokens=regexp_tokenize(sen, "[\w']+") #tekonize words 
    return tokens


# In[21]:


#example of normalizer application on our Data
DF['normalized'] = DF.Content.apply(normalizer)
DF['normalized'][1]


# In[22]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# create a pipeline that will automate modeling workflow: apply normalizer function+create Bag of Words(BoW)+use algorithm of Naive Bayes:
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=normalizer)),  # normalized strings to token integer counts
    ('classifier', MultinomialNB()) # train on vectors w/ Naive Bayes classifier
])


# In[23]:


import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DF['Content'],DF['Topic'], test_size=0.33, random_state=42)


# In[24]:


pipeline.fit(X_train,y_train)


# In[25]:


from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import f1_score
y_pred= pipeline.predict(X_test)
#let's see our classification report
#print(classification_report(y_test, y_pred, target_names=list(DF.Topic.unique())))


# In[26]:


#let's see our confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt     
# cm=confusion_matrix(y_test, y_pred, labels=['Anthropology','Astronomy'])
# ax= plt.subplot()
# sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# # labels, title and ticks
# ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
# ax.set_title('Confusion Matrix'); 
# # ax.xaxis.set_ticklabels(['Anthropology','Astronomy']); ax.yaxis.set_ticklabels(['Anthropology','Astronomy']);


# In[27]:


# #let's test our algorithm:
# mytext='إن تباين مفهوم (علم الكون) وفقًا لحجم كتابة أول حرف من تهجئته، يوحي بأن هناك أكثر من كون تم إيجاده في هذه الحياة، وهو أمر مستحيل بالطبيعة، فلا يوجد في هذه الحياة سوى كون واحد، وذلك يضعنا في مشكلة كيفية إمكان تطبيق تجارب مختلفة لأكثر من مفهوم على كون واحد'
# myprediction= pipeline.predict([mytext])
# print(myprediction[0])


# In[28]:


import pickle
pickle.dump(pipeline,open('model.pkl',"wb"))

