{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Sentiment Analysis on Playstore reviews\n",
    "\n",
    "The aim of this project is to classify playstore reviews as positive or negative..Developers often find it difficult\n",
    "understanding and finding out how and where to improve the applications functionality as rating alone does not provide any\n",
    "useful information regarding this matter.This is where analyzing textual comments would come to a great benefit.The proposed model can detect positive or negative comments automatecially and later can be clustered to find out which things that users are liking and which can/should be improved.<br>\n",
    "\n",
    "The Dataset was collected from <a href=\"https://www.kaggle.com/lava18/google-play-store-apps\">Play store App Review</a>\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# importing all the necessary packages\n",
    "importing all the necessary packages.<br>Pandas is a high performance and easy to use data analysis tool.<br>Numpy for using multi dimensional arrays.<br>NLTK stands for natural language processing.It is a must have tool for language processing.Keras is a deep learning library which makes using deep learing model so easy.<br>Then there is tokenizer that will tokenize every word and pad_sequences is used to ensure that all sequences in a list have the same length.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import nltk\n",
    "import keras\n",
    "from keras.preprocessing.text import Tokenizer\n",
    "from keras.preprocessing.sequence import pad_sequences"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Opening the dataset then Editing\n",
    "\n",
    "We will convert the csv file as pandas dataframe for easier execution.\n",
    "dropna() method allows to analyze and drop Rows/Columns with Null values in the dataframe\n",
    "then we will replace the sentiment labels to numerical value as you know It is easier for computers to read numbers 😁.<br> Our dataset contains three lables.For easier demonstration,we will consider binary classifiaction so we will consider neutarl labelling as positive and hence will give the below 1 for positive and 0 for negative respectively"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df = pd.read_csv(\"googleplaystore_user_reviews.csv\", na_values=\"nan\")\n",
    "df = df.dropna(subset=['App','Translated_Review','Sentiment'], how='any')\n",
    "df['Sentiment'] = df['Sentiment'].replace(['Positive'],'1')\n",
    "df['Sentiment'] = df['Sentiment'].replace(['Negative'],'0')\n",
    "df['Sentiment'] = df['Sentiment'].replace(['Neutral'],'1')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "# importing some other necessary packages\n",
    "In this step we will import word_tokenize and stopwords from.Stopword are the words like am,are,i,my,is etc. They are pretty common and does not contain important significance thus they can/should be removed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import string\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.corpus import stopwords"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# reviews are categorized as lines\n",
    "In this step we will convert the reviews in our dataset to list.We can see we have 37427 reviews."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "37427\n"
     ]
    }
   ],
   "source": [
    "review_lines = list()\n",
    "lines = df['Translated_Review'].values.tolist()\n",
    "print (len(lines))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# tokenization and removing punctuation and stop words\n",
    "\n",
    "We will tokenize the word from every line or reviwe.Then we will do some preprocessing.Well,preprocessing is the process of cleaning or preparing a dataset or set of data for a model.It generally depends on one's personal need as how he/she would preprocess the dataset.But stop_words removing,punctuation removing,numerical value removig are some of the well known preprocessing techniques. Here we will first make all the words lowercase then we will remove punctuation after that we will remove the stop_words. There are also some other well know tecniques for preprocessing like stemming and lemmatization.You can give them a try.<br>\n",
    "<b>Remember</b> preprocessing is a very necessary steps for any machine learning model.There is a saying \"garbage in,garbage out\".Your input will be the key to get the desired output."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "37427"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for line in lines :\n",
    "    tokens = word_tokenize(line)\n",
    "    tokens = [w.lower() for w in tokens]\n",
    "    table =str.maketrans('','',string.punctuation)\n",
    "    stripped = [w.translate(table) for w in tokens]\n",
    "    words = [word for word in stripped if word.isalpha()]\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    words = [w for w in words if not w in stop_words]\n",
    "    review_lines.append(words)\n",
    "len(review_lines)\n",
    "#print(review_lines)\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# word2vec model\n",
    "\n",
    "Now comes the most important part. Word2Vec is one of the most popular technique to learn word embeddings.Now what is word embedding???<br>\n",
    "\n",
    "Well,Word embedding is one of the most popular representation of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc.<br><br>\n",
    "\n",
    "See here <a href=\"https://medium.com/explore-artificial-intelligence/word2vec-a-baby-step-in-deep-learning-but-a-giant-leap-towards-natural-language-processing-40fe4e8602ba\">Word2Vec</a> for more info...<br><br>\n",
    "\n",
    "We will import Word2vec model from genism.Then we will initialize the model.<br>\n",
    "We will pass our list (review lines ).Word vector Dimensionality has been given as 100.One can use any dimension s/he needs.But 100 recommended.Window size has been given as 5 which means Maximum distance between the current and predicted word within a sentence should be 5. Workers define how many threads will be used.Generally,Training will be fastrer with multicore machines.\n",
    "\n",
    "For more information.... <a href =\"https://radimrehurek.com/gensim/models/word2vec.html\">Genism Word2vec</a>\n",
    "\n",
    "Then we will print how many words we have got in our vocabulary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "G:\\Anaconda\\lib\\site-packages\\gensim\\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial\n",
      "  warnings.warn(\"detected Windows; aliasing chunkize to chunkize_serial\")\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "total word: 21481\n"
     ]
    }
   ],
   "source": [
    "import gensim\n",
    "\n",
    "model = gensim.models.Word2Vec(sentences=review_lines,size=100,window = 5,workers =4,min_count=1)\n",
    "words = list(model.wv.vocab)\n",
    "print('total word: %d' %len(words))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# saving the model\n",
    "\n",
    "We will save the model for later use.We do not need to train word embeddings.By saving we can easily use the model later"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = 'r.txt'\n",
    "model.wv.save_word2vec_format(filename,binary=False)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Mapping words and their corrosponding vectors\n",
    "\n",
    "In this step first an empty dictionary is initialized then the word2vec model is read.For each word their corresponding vector is mapped from word2Vec model \n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "embeddings_index = {}\n",
    "f = open(os.path.join('','r.txt'),encoding = \"utf-8\")\n",
    "for line in f:\n",
    "    values = line.split()\n",
    "    word = values[0]\n",
    "    coefs = np.asarray(values[1:])\n",
    "    embeddings_index[word]=coefs\n",
    "f.close()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Applying tokenizer \n",
    "\n",
    "Tokenizer and pad_sequence is applied on the review sentences."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "found 21481 unique tokens \n",
      "Shape of review  (37427, 100)\n",
      "shape of senti (37427,)\n"
     ]
    }
   ],
   "source": [
    "tk = Tokenizer()\n",
    "tk.fit_on_texts(review_lines)  #list of texts to train on\n",
    "sequences = tk.texts_to_sequences(review_lines)  # list of texts to turn to sequences.\n",
    "word_index = tk.word_index\n",
    "print(\"found %s unique tokens \" % len(word_index))\n",
    "review_pad = pad_sequences(sequences,maxlen=100)  #ensureing all sequences in a list to have the same length.\n",
    "sentiment = df['Sentiment'].values  \n",
    "print('Shape of review ', review_pad.shape)\n",
    "print('shape of senti' , sentiment.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# map word embeddings from the loaded word2vec model for each word in our word_index items\n",
    "\n",
    "\n",
    "First, total number of word is initialized.It has to be (1+total word) as it starts from zero index.Then using numpy a (21482*100) size matrix full of zeroes has been created. And for every word their correspondind vector value is matched\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "21482\n"
     ]
    }
   ],
   "source": [
    "num_words = len(word_index) + 1\n",
    "embedd = np.zeros((num_words,100))\n",
    "\n",
    "for word , i in word_index.items():\n",
    "    if i > num_words:\n",
    "        continue\n",
    "    embedd_vec = embeddings_index.get(word)\n",
    "    if embedd_vec is not None:\n",
    "        embedd[i] = embedd_vec     \n",
    "print(num_words)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# RNN model \n",
    "\n",
    "In this step,we will create our RNN model.Keras sequential model is used here and on the first layer we will feed the word embeddings as the embedding layer.<br>\n",
    "Next, GRU-an RNN architecture, which is similar to LSTM will be used.Here,both dropout and recurrent_dropout is used for to drop for the linear transformation of the inputs and to drop for the linear transformation of the recurrent state respectively.<br>\n",
    "For output layer Dense layer has been used and units has been set to 1 as our model is to predict binary classification.Sigmoid is used as an activation function.<br>\n",
    "for calculating loss 'binary_crossentropy' has been used. It's a method of evaluating how well specific algorithm models the given data.There are some other loss function as well.But for binary classifiaction 'binary_crossentropy' is the suitable one.<br>\n",
    "Adam has been used as the optimizer.The function of optimizer is to minimize the loss.There are also so many optimizers available.You can use any to see which fits well.<br>\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "embedding_2 (Embedding)      (None, 100, 100)          2148200   \n",
      "_________________________________________________________________\n",
      "gru_2 (GRU)                  (None, 32)                12768     \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 1)                 33        \n",
      "=================================================================\n",
      "Total params: 2,161,001\n",
      "Trainable params: 12,801\n",
      "Non-trainable params: 2,148,200\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense,GRU\n",
    "from keras.layers.embeddings import Embedding\n",
    "from keras.initializers import constant\n",
    "\n",
    "model = Sequential()\n",
    "embedding_layer =  Embedding(num_words,100,embeddings_initializer= constant(embedd),input_length=100,trainable=False)\n",
    "model.add(embedding_layer)\n",
    "model.add(GRU(units=32, dropout=0.2,recurrent_dropout=0.2))\n",
    "model.add(Dense(1,activation='sigmoid'))\n",
    "model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Creating Training and Testing set\n",
    "\n",
    "We will split our dataset into two set.One for trainig and other for testing or validation.20% has been randomly selceted for validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "shape of X_train_pad  (29942, 100)\n",
      "shape of y_train  (29942,)\n",
      "shape of X_test_pad  (7485, 100)\n",
      "shape of y_test  (7485,)\n"
     ]
    }
   ],
   "source": [
    "\n",
    "VALIDATION_SPLIT = 0.2\n",
    "\n",
    "indices = np.arange(review_pad.shape[0])\n",
    "np.random.shuffle(indices)\n",
    "review_pad = review_pad[indices]\n",
    "sentiment = sentiment[indices]\n",
    "num_validation = int (VALIDATION_SPLIT * review_pad.shape[0])\n",
    "\n",
    "X_train_pad = review_pad[:-num_validation]\n",
    "y_train = sentiment[:-num_validation]\n",
    "X_test_pad = review_pad[-num_validation:]\n",
    "y_test = sentiment[-num_validation:]\n",
    "\n",
    "\n",
    "\n",
    "print('shape of X_train_pad ', X_train_pad.shape)\n",
    "print('shape of y_train ', y_train.shape)\n",
    "\n",
    "print('shape of X_test_pad ', X_test_pad.shape)\n",
    "print('shape of y_test ', y_test.shape)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# training the classification model on train set and validating on validation set\n",
    "\n",
    "10 epochs has been given for demonstration purpose"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From G:\\Anaconda\\lib\\site-packages\\tensorflow\\python\\ops\\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use tf.cast instead.\n",
      "Train on 29942 samples, validate on 7485 samples\n",
      "Epoch 1/10\n",
      " - 31s - loss: 0.4459 - acc: 0.7916 - val_loss: 0.3848 - val_acc: 0.8179\n",
      "Epoch 2/10\n",
      " - 32s - loss: 0.3831 - acc: 0.8207 - val_loss: 0.3651 - val_acc: 0.8283\n",
      "Epoch 3/10\n",
      " - 32s - loss: 0.3699 - acc: 0.8277 - val_loss: 0.3509 - val_acc: 0.8411\n",
      "Epoch 4/10\n",
      " - 33s - loss: 0.3588 - acc: 0.8355 - val_loss: 0.3407 - val_acc: 0.8445\n",
      "Epoch 5/10\n",
      " - 33s - loss: 0.3498 - acc: 0.8402 - val_loss: 0.3352 - val_acc: 0.8485\n",
      "Epoch 6/10\n",
      " - 34s - loss: 0.3446 - acc: 0.8440 - val_loss: 0.3347 - val_acc: 0.8473\n",
      "Epoch 7/10\n",
      " - 35s - loss: 0.3371 - acc: 0.8479 - val_loss: 0.3286 - val_acc: 0.8478\n",
      "Epoch 8/10\n",
      " - 36s - loss: 0.3335 - acc: 0.8508 - val_loss: 0.3196 - val_acc: 0.8542\n",
      "Epoch 9/10\n",
      " - 38s - loss: 0.3286 - acc: 0.8522 - val_loss: 0.3157 - val_acc: 0.8569\n",
      "Epoch 10/10\n",
      " - 38s - loss: 0.3241 - acc: 0.8567 - val_loss: 0.3136 - val_acc: 0.8577\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x2af1e2e24a8>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_pad,y_train,batch_size=64,epochs=10,validation_data= (X_test_pad,y_test),verbose=2)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Evaluting the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Testing Accuracy:  0.8577\n"
     ]
    }
   ],
   "source": [
    "score = model.evaluate(X_test_pad, y_test, verbose=0)\n",
    "print(\"Testing Accuracy:  {:.4f}\".format(accuracy))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# printing accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 85.77%\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy: %.2f%%\" % (score[1]*100))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Testing sample dataset\n",
    "\n",
    "Here we will give there sample test dataset to see whether our model can predict the label of them.If the predicted value is closer to 1 then the review or comment will be positive,if the predicted value is closer to 0 then it will be a negative review"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.93707865],\n",
       "       [0.2900746 ],\n",
       "       [0.09323388]], dtype=float32)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_sample1=\"just loving it\"\n",
    "test_sample2=\"I hate using this button,please fix it\"\n",
    "test_sample3=\"this app is bad \"\n",
    "\n",
    "\n",
    "test_samples = [test_sample1,test_sample2,test_sample3]\n",
    "test_samples_tokens = tk.texts_to_sequences(test_samples)\n",
    "\n",
    "pad =pad_sequences(test_samples_tokens,maxlen=100)\n",
    "\n",
    "model.predict(x =pad)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Here we can see that our model has given a good result as it can classify the reviews as positive and negative."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
