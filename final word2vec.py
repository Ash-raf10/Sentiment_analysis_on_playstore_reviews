#!/usr/bin/env python
# coding: utf-8

# # importing all the necessary packages

# In[1]:


import pandas as pd
import numpy as np
import nltk
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# # read csv dataset then edit

# In[2]:



df = pd.read_csv("googleplaystore_user_reviews.csv", na_values="nan")
df = df.dropna(subset=['App','Translated_Review','Sentiment'], how='any')
df['Sentiment'] = df['Sentiment'].replace(['Positive'],'1')
df['Sentiment'] = df['Sentiment'].replace(['Negative'],'0')
df['Sentiment'] = df['Sentiment'].replace(['Neutral'],'1')


# In[3]:


import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# # reviews are categorized as lines

# In[4]:


review_lines = list()
lines = df['Translated_Review'].values.tolist()
print (len(lines))


# # tokenization and removing punctuation and stop words

# In[5]:


for line in lines :
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table =str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    review_lines.append(words)
len(review_lines)
#print(review_lines)
    


# # word2vec model

# In[6]:


import gensim

model = gensim.models.Word2Vec(sentences=review_lines,size=100,window = 5,workers =4,min_count=1)
words = list(model.wv.vocab)
print('total word: %d' %len(words))


# # saving the model

# In[7]:


filename = 'r.txt'
model.wv.save_word2vec_format(filename,binary=False)


# # word embedding as a directory of words to vectors

# In[8]:


import os
embeddings_index = {}
f = open(os.path.join('','r.txt'),encoding = "utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word]=coefs
f.close()


# # converting the word embedding into tokenized vector

# In[9]:


tk = Tokenizer()
tk.fit_on_texts(review_lines)
sequences = tk.texts_to_sequences(review_lines)
word_index = tk.word_index
print("found %s unique tokens " % len(word_index))
review_pad = pad_sequences(sequences,maxlen=100)
sentiment = df['Sentiment'].values
print('Shape of review ', review_pad.shape)
print('shape of senti' , sentiment.shape)


# # map embeddings from the loaded word2vec model for each word 

# In[10]:


num_words = len(word_index) + 1
embedd = np.zeros((num_words,100))

for word , i in word_index.items():
   if i > num_words:
       continue
   embedd_vec = embeddings_index.get(word)
   if embedd_vec is not None:
       embedd[i] = embedd_vec     
print(num_words)


# # embedding matrix as input to the Embedding layer

# In[11]:



from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.initializers import constant

model = Sequential()
embedding_layer =  Embedding(num_words,100,embeddings_initializer= constant(embedd),input_length=100,trainable=False)
model.add(embedding_layer)
model.add(GRU(units=32, dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# # training the sentiment classification model

# In[12]:



VALIDATION_SPLIT = 0.2

indices = np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad = review_pad[indices]
sentiment = sentiment[indices]
num_validation = int (VALIDATION_SPLIT * review_pad.shape[0])

X_train_pad = review_pad[:-num_validation]
y_train = sentiment[:-num_validation]
X_test_pad = review_pad[-num_validation:]
y_test = sentiment[-num_validation:]



print('shape of X_train_pad ', X_train_pad.shape)
print('shape of y_train ', y_train.shape)

print('shape of X_test_pad ', X_test_pad.shape)
print('shape of y_train ', y_test.shape)


# # training the classification model on train and validation test set

# In[13]:


model.fit(X_train_pad,y_train,batch_size=64,epochs=10,validation_data= (X_test_pad,y_test),verbose=2)
scores = model.evaluate(X_test_pad, y_test, verbose=0)


# # printing accuracy

# In[14]:


print("Accuracy: %.2f%%" % (scores[1]*100))


# # Testing sample dataset

# In[15]:


test_sample1="just loving it"
test_sample2="I hate using this button,please fix it"
test_sample3="this app is bad "


test_samples = [test_sample1,test_sample2,test_sample3]
test_samples_tokens = tk.texts_to_sequences(test_samples)

pad =pad_sequences(test_samples_tokens,maxlen=100)

model.predict(x =pad)


# In[ ]:




