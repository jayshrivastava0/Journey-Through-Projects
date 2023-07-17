#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sounddevice as sdd


# In[2]:


import scipy


# In[3]:


from scipy.io.wavfile import write


# In[4]:


fs=44100


# In[5]:


second=float(input("enter the time duration in second : \n"))


# In[6]:


print('recording...\n')


# In[ ]:





# In[7]:


record_voice=sdd.rec(int(second * fs),samplerate=fs,channels=2)


# In[8]:


sdd.wait()


# In[9]:


write('recording.wav',fs,record_voice)


# In[10]:


print('Finished...\nPlease Check the folder for recording...')


# In[ ]:




