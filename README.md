
# Sentiment Analysis on Playstore reviews

The aim of this project is to classify playstore reviews as positive or negative..Developers often find it difficult
understanding and finding out how and where to improve the applications functionality as rating alone does not provide any
useful information regarding this matter.This is where analyzing textual comments would come to a great benefit.The proposed model can detect positive or negative comments automatecially and later can be clustered to find out which things that users are liking and which can/should be improved.<br>

The Dataset was collected from <a href="https://www.kaggle.com/lava18/google-play-store-apps">Play store App Review</a>



# importing all the necessary packages
importing all the necessary packages.<br>Pandas is a high performance and easy to use data analysis tool.<br>Numpy for using multi dimensional arrays.<br>NLTK stands for natural language processing.It is a must have tool for language processing.Keras is a deep learning library which makes using deep learing model so easy.<br>Then there is tokenizer that will tokenize every word and pad_sequences is used to ensure that all sequences in a list have the same length.



```python
import pandas as pd
import numpy as np
import nltk
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
```

    Using TensorFlow backend.
    

# Opening the dataset then Editing

We will convert the csv file as pandas dataframe for easier execution.
dropna() method allows to analyze and drop Rows/Columns with Null values in the dataframe
then we will replace the sentiment labels to numerical value as you know It is easier for computers to read numbers 😁.<br> Our dataset contains three lables.For easier demonstration,we will consider binary classifiaction so we will consider neutarl labelling as positive and hence will give the below 1 for positive and 0 for negative respectively


```python

df = pd.read_csv("googleplaystore_user_reviews.csv", na_values="nan")
df = df.dropna(subset=['App','Translated_Review','Sentiment'], how='any')
df['Sentiment'] = df['Sentiment'].replace(['Positive'],'1')
df['Sentiment'] = df['Sentiment'].replace(['Negative'],'0')
df['Sentiment'] = df['Sentiment'].replace(['Neutral'],'1')

```


# importing some other necessary packages
In this step we will import word_tokenize and stopwords from.Stopword are the words like am,are,i,my,is etc. They are pretty common and does not contain important significance thus they can/should be removed.


```python
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
```

# reviews are categorized as lines
In this step we will convert the reviews in our dataset to list.We can see we have 37427 reviews.


```python
review_lines = list()
lines = df['Translated_Review'].values.tolist()
print (len(lines))
```

    37427
    

# tokenization and removing punctuation and stop words

We will tokenize the word from every line or reviwe.Then we will do some preprocessing.Well,preprocessing is the process of cleaning or preparing a dataset or set of data for a model.It generally depends on one's personal need as how he/she would preprocess the dataset.But stop_words removing,punctuation removing,numerical value removig are some of the well known preprocessing techniques. Here we will first make all the words lowercase then we will remove punctuation after that we will remove the stop_words. There are also some other well know tecniques for preprocessing like stemming and lemmatization.You can give them a try.<br>
<b>Remember</b> preprocessing is a very necessary steps for any machine learning model.There is a saying "garbage in,garbage out".Your input will be the key to get the desired output.


```python
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
    
```




    37427



# word2vec model

Now comes the most important part. Word2Vec is one of the most popular technique to learn word embeddings.Now what is word embedding???<br>

Well,Word embedding is one of the most popular representation of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc.<br><br>

See here <a href="https://medium.com/explore-artificial-intelligence/word2vec-a-baby-step-in-deep-learning-but-a-giant-leap-towards-natural-language-processing-40fe4e8602ba">Word2Vec</a> for more info...<br><br>

We will import Word2vec model from genism.Then we will initialize the model.<br>
We will pass our list (review lines ).Word vector Dimensionality has been given as 100.One can use any dimension s/he needs.But 100 recommended.Window size has been given as 5 which means Maximum distance between the current and predicted word within a sentence should be 5. Workers define how many threads will be used.Generally,Training will be fastrer with multicore machines.

For more information.... <a href ="https://radimrehurek.com/gensim/models/word2vec.html">Genism Word2vec</a>

Then we will print how many words we have got in our vocabulary


```python
import gensim

model = gensim.models.Word2Vec(sentences=review_lines,size=100,window = 5,workers =4,min_count=1)
words = list(model.wv.vocab)
print('total word: %d' %len(words))
```

    G:\Anaconda\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    

    total word: 21481
    

# saving the model

We will save the model for later use.We do not need to train word embeddings.By saving we can easily use the model later


```python
filename = 'r.txt'
model.wv.save_word2vec_format(filename,binary=False)


```

# Mapping words and their corrosponding vectors

In this step first an empty dictionary is initialized then the word2vec model is read.For each word their corresponding vector is mapped from word2Vec model 





```python
import os
embeddings_index = {}
f = open(os.path.join('','r.txt'),encoding = "utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word]=coefs
f.close()

```

# Applying tokenizer 

Tokenizer and pad_sequence is applied on the review sentences.


```python
tk = Tokenizer()
tk.fit_on_texts(review_lines)  #list of texts to train on
sequences = tk.texts_to_sequences(review_lines)  # list of texts to turn to sequences.
word_index = tk.word_index
print("found %s unique tokens " % len(word_index))
review_pad = pad_sequences(sequences,maxlen=100)  #ensureing all sequences in a list to have the same length.
sentiment = df['Sentiment'].values  
print('Shape of review ', review_pad.shape)
print('shape of senti' , sentiment.shape)
```

    found 21481 unique tokens 
    Shape of review  (37427, 100)
    shape of senti (37427,)
    

# map word embeddings from the loaded word2vec model for each word in our word_index items


First, total number of word is initialized.It has to be (1+total word) as it starts from zero index.Then using numpy a (21482*100) size matrix full of zeroes has been created. And for every word their correspondind vector value is matched



```python
num_words = len(word_index) + 1
embedd = np.zeros((num_words,100))

for word , i in word_index.items():
    if i > num_words:
        continue
    embedd_vec = embeddings_index.get(word)
    if embedd_vec is not None:
        embedd[i] = embedd_vec     
print(num_words)
```

    21482
    

# RNN model 

In this step,we will create our RNN model.Keras sequential model is used here and on the first layer we will feed the word embeddings as the embedding layer.<br>
Next, GRU-an RNN architecture, which is similar to LSTM will be used.Here,both dropout and recurrent_dropout is used for to drop for the linear transformation of the inputs and to drop for the linear transformation of the recurrent state respectively.<br>
For output layer Dense layer has been used and units has been set to 1 as our model is to predict binary classification.Sigmoid is used as an activation function.<br>
for calculating loss 'binary_crossentropy' has been used. It's a method of evaluating how well specific algorithm models the given data.There are some other loss function as well.But for binary classifiaction 'binary_crossentropy' is the suitable one.<br>
Adam has been used as the optimizer.The function of optimizer is to minimize the loss.There are also so many optimizers available.You can use any to see which fits well.<br>




```python

from keras.models import Sequential
from keras.layers import Dense,GRU
from keras.layers.embeddings import Embedding
from keras.initializers import constant

model = Sequential()
embedding_layer =  Embedding(num_words,100,embeddings_initializer= constant(embedd),input_length=100,trainable=False)
model.add(embedding_layer)
model.add(GRU(units=32, dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 100, 100)          2148200   
    _________________________________________________________________
    gru_2 (GRU)                  (None, 32)                12768     
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 2,161,001
    Trainable params: 12,801
    Non-trainable params: 2,148,200
    _________________________________________________________________
    

# Creating Training and Testing set

We will split our dataset into two set.One for trainig and other for testing or validation.20% has been randomly selceted for validation


```python

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
print('shape of y_test ', y_test.shape)

```

    shape of X_train_pad  (29942, 100)
    shape of y_train  (29942,)
    shape of X_test_pad  (7485, 100)
    shape of y_test  (7485,)
    

# training the classification model on train set and validating on validation set

10 epochs has been given for demonstration purpose


```python
model.fit(X_train_pad,y_train,batch_size=64,epochs=10,validation_data= (X_test_pad,y_test),verbose=2)


```

    WARNING:tensorflow:From G:\Anaconda\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Train on 29942 samples, validate on 7485 samples
    Epoch 1/10
     - 31s - loss: 0.4459 - acc: 0.7916 - val_loss: 0.3848 - val_acc: 0.8179
    Epoch 2/10
     - 32s - loss: 0.3831 - acc: 0.8207 - val_loss: 0.3651 - val_acc: 0.8283
    Epoch 3/10
     - 32s - loss: 0.3699 - acc: 0.8277 - val_loss: 0.3509 - val_acc: 0.8411
    Epoch 4/10
     - 33s - loss: 0.3588 - acc: 0.8355 - val_loss: 0.3407 - val_acc: 0.8445
    Epoch 5/10
     - 33s - loss: 0.3498 - acc: 0.8402 - val_loss: 0.3352 - val_acc: 0.8485
    Epoch 6/10
     - 34s - loss: 0.3446 - acc: 0.8440 - val_loss: 0.3347 - val_acc: 0.8473
    Epoch 7/10
     - 35s - loss: 0.3371 - acc: 0.8479 - val_loss: 0.3286 - val_acc: 0.8478
    Epoch 8/10
     - 36s - loss: 0.3335 - acc: 0.8508 - val_loss: 0.3196 - val_acc: 0.8542
    Epoch 9/10
     - 38s - loss: 0.3286 - acc: 0.8522 - val_loss: 0.3157 - val_acc: 0.8569
    Epoch 10/10
     - 38s - loss: 0.3241 - acc: 0.8567 - val_loss: 0.3136 - val_acc: 0.8577
    




    <keras.callbacks.History at 0x2af1e2e24a8>



# Evaluting the model


```python
score = model.evaluate(X_test_pad, y_test, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy))
```

    Testing Accuracy:  0.8577
    

# printing accuracy


```python
print("Accuracy: %.2f%%" % (score[1]*100))
```

    Accuracy: 85.77%
    

# Testing sample dataset

Here we will give there sample test dataset to see whether our model can predict the label of them.If the predicted value is closer to 1 then the review or comment will be positive,if the predicted value is closer to 0 then it will be a negative review


```python
test_sample1="just loving it"
test_sample2="I hate using this button,please fix it"
test_sample3="this app is bad "


test_samples = [test_sample1,test_sample2,test_sample3]
test_samples_tokens = tk.texts_to_sequences(test_samples)

pad =pad_sequences(test_samples_tokens,maxlen=100)

model.predict(x =pad)

```




    array([[0.93707865],
           [0.2900746 ],
           [0.09323388]], dtype=float32)



# Here we can see that our model has given a good result as it can classify the reviews as positive and negative.


```python

```
