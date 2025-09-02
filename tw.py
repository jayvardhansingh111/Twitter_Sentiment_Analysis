import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')
# print(stopwords.words('english'))

# dataset=pd.read_csv('twitter_training.csv')
# dataset=pd.read_csv('twitter_pbl.csv', encoding= 'ISO-8859-1')
dataset = pd.read_csv('D:/TW_/twitter_pbl.csv', encoding='ISO-8859-1')
print(dataset.head())

col_names = ['target' , 'id' , 'date' , 'flag' , 'user' , 'text']
dataset.columns = col_names
print(dataset.shape)

dataset.isnull().sum()

# Distribution of tweets
dataset['target'].value_counts()

#  Converting 0 to -ve and 4 to +ve
dataset['target'] = dataset['target'].map({0:0 , 4:1})
dataset['target'].value_counts()

# # Stemming
stremmer = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content) # removing not a-z and A-Z
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stremmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# dataset['text'] = dataset['text'].apply(stemming)
# dataset.head()

x = dataset['text']
y = dataset['target']

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 0)

# convert textual data to numerical data
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
print(x_train)  # show the trian data

# Training the model
model = LogisticRegression()
model.fit(x_train , y_train)

# Testing the model
y_pred = model.predict(x_test)
# print(accuracy_score(y_test , y_pred))

# Function to predict the sentiment
def predict_sentiment(text):
    text = re.sub('[^a-zA-Z]',' ',text) # removing not a-z and A-Z
    text = text.lower()
    text = text.split()
    text = [stremmer.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    sentiment = model.predict(text)
    if sentiment == 0:
        return "Negative"
    else:
        return "Positive"

# Testing the model
print(predict_sentiment("I hate you"))
print(predict_sentiment("I love you"))

# Save the model
import pickle
pickle.dump(model , open('model.pkl' , 'wb'))

pickle.dump(vectorizer , open('vectorizer.pkl' , 'wb'))

