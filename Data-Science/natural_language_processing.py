#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter


# In[2]:


from nltk.tokenize import sent_tokenize


# In[3]:


text= 'let us checkout how sentence tokenizing work. This would be the second sentence. If everything works fine, this would be the third sentence'


# In[4]:


print(sent_tokenize(text))


# In[5]:


text1='This will lead to a problem e.g. here'
print(sent_tokenize(text1))


# In[6]:


text2='if we use the abbreviation for number no. , it will lead to new sentence'
print(sent_tokenize(text2))


# In[7]:


import spacy


# In[8]:


nlp=spacy.load('en_core_web_sm')


# In[9]:


doc=nlp('this will lead to a problem, e.g. here.')


# In[10]:


for sent in doc.sents:
    print('New sentence: ')
    print(sent.text)


# In[11]:


import requests


# In[12]:


from bs4 import BeautifulSoup


# In[13]:


import re


# In[14]:


request = requests.get('https://en.wikipedia.org/wiki/Car')


# In[15]:


scrapped_html=request.content


# In[16]:


soup=BeautifulSoup(scrapped_html,'html.parser')


# In[45]:


text=soup.find_all(text=True)
#text


# In[18]:


tags = re.compile(r'<[^>]+>')
def remove_tags(text):
    return tags.sub('',text)


# In[19]:


def preprocess_text(sen):
    # removing html tags
    sentence=remove_tags(sen)
    #remove punctuations and numbers
    sentence=re.sub('[^a-zA-Z]','',sentence)
    # single character removal
    sentence = re.sub(r'\s+[a+zA-Z]\s+','',sentence)
    # Removing multiple spaces
    sentences = re.sub(r'\s+','',sentence)
    return sentence


# In[20]:


cleaned_sentences = []
for sentence in text :
    cleaned_sentences.append(preprocess_text(sentence))
cleaned_sentences=[sentence for sentence in cleaned_sentences if sentence != '']


# In[21]:


#cleaned_sentences


# In[22]:


#pip install pyspellchecker
from spellchecker import SpellChecker
spell = SpellChecker()


# In[23]:


# find those words that may be misspelled
check_sentence = spell.unknown(['Maybe','there','are','some','wrong','words','in', 'heere'])


# In[24]:


for word in check_sentence:
    # Get the wrong words
    print(spell.correction(word))


# In[25]:


def decontracted(phrase):
    # define specific phrases
    phrase = re.sub(r"won\'t",'will not', phrase)
    phrase = re.sub(r"can\'t", 'can not', phrase)
    
    # general
    phrase = re.sub(r"n\'t", ' not', phrase) #aren't
    phrase = re.sub(r"\'re",' are',phrase) #we're
    phrase = re.sub("\'s",' is', phrase) #he's
    phrase = re.sub(r"\'d", ' would', phrase) #she'd
    phrase = re.sub(r"\'ll", ' will', phrase)#you'll
    phrase = re.sub(r"\'ve", ' have', phrase) #you've
    phrase = re.sub(r"\'m", ' am', phrase) #I'm
    return phrase


# In[26]:


test_sentence = "Hey I'm full of contractions, I'd bet, you can't clean me"
print(decontracted(test_sentence))


# In[27]:


#stemming
import nltk
from nltk.stem.porter import *


# In[28]:


porter_stemmer = PorterStemmer()


# In[29]:


sentence = "This is the particular sentence for porter's Stemming Algorithm"


# In[30]:


tokenized_words = nltk.word_tokenize(sentence)


# In[31]:


StemWords=[porter_stemmer.stem(word) for word in tokenized_words]


# In[32]:


print(' '.join(StemWords))


# In[33]:


porter_stemmer.stem('alumnus')


# In[34]:


porter_stemmer.stem('Alumni')


# In[35]:


porter_stemmer.stem("ALumnae")


# In[36]:


#lemmetiation


# In[37]:


from nltk.stem import WordNetLemmatizer


# In[38]:


nltk.download('wordnet')


# In[39]:


lemmatizer = WordNetLemmatizer()


# In[40]:


nltk.download('omw-1.4')


# In[ ]:





# In[ ]:





# In[41]:


print('likes: ', lemmatizer.lemmatize('likes'))


# In[42]:


print("better :", lemmatizer.lemmatize('better'))


# In[43]:


print('better:', lemmatizer.lemmatize('better', pos = 'a'))


# In[ ]:




