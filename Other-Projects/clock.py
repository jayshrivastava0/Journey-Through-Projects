#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter


# In[2]:


from tkinter import Label, Tk


# In[3]:


import time


# In[4]:


app_window = Tk()


# In[5]:


app_window.title('Digital Clock')


# In[6]:


app_window.geometry("450x150")


# In[ ]:





# In[7]:


text_font=('Boulder',68,'bold')


# In[8]:


background = '#f56565'


# In[9]:


foreground = '#300000'


# In[10]:


border_width = 50


# In[11]:


label = Label(app_window, font = text_font, bg = background, fg = foreground, bd = border_width)


# In[12]:


label.grid(row=10, column =1)


# In[13]:


def digital_clock():
    time_live = time.strftime("%H:%M:%S")
    label.config(text=time_live)
    label.after(200,digital_clock)


# In[14]:


digital_clock()


# In[15]:


app_window.mainloop()


# In[ ]:




