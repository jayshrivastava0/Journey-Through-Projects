#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime


# In[2]:


current_date = datetime.date.today().strftime('%Y-%m-%d')


# In[3]:


current_date


# In[4]:


hello = current_date.split('-')


# In[5]:


hello


# In[6]:


bday_log = [
    ('Aman',('1999','10','19'))]


# In[7]:


add = input('To add birthday type y:\n').strip().lower()


# In[8]:


if add [:1] == 'y':
    new = input('Add birthday in format yyyy-mm-dd:\n')
    # print(new_lst)
    name = input('Whose bday?\n')
    date = new.split('-')


# In[9]:


date1=tuple(date)


# In[10]:


bday_log.append(name)
bday_log.append(date1)


# In[11]:


bday_log


# In[ ]:





# In[12]:


date1


# In[13]:


date=list(date1)


# In[14]:


date


# In[15]:


birth_year=int(date[0])
birth_month=int(date[1])
birth_date=int(date[2])


# In[16]:


type(birth_year)


# In[17]:


current_year=int(hello[0])
current_month=int(hello[1])
current_date=int(hello[2])


# In[18]:


current_date


# In[19]:


from datetime import date


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


all_current=datetime.date(current_year,current_month,current_date)


# In[21]:


all_current


# In[22]:


birth123=datetime.date(birth_year,birth_month,birth_date)


# In[23]:


birth123


# In[24]:


def numOfDays(all_current, birth123):
    return (all_current-birth123).days


# In[25]:


numOfDays(all_current, birth123)


# In[26]:


year = numOfDays(all_current, birth123)//365.26
year


# In[27]:


left = int(numOfDays(all_current, birth123) - year*365.26)
left


# In[28]:


print(f"{left} days left till {name}'s birthday")


# In[29]:


import pywhatkit


# In[30]:


#send minute
send_min=int(str(datetime.datetime.now()).split()[1].split(':')[1])+2
send_min


# In[31]:


#sendhour
send_hour=int(str(datetime.datetime.now()).split()[1].split(':')[0])
send_hour


# In[32]:


phone1= input('What is your phone number, that you want the reminder sent to??\n')


# In[33]:


phone='+91'+phone1


# In[34]:


phone


# In[35]:


if current_month == birth_month and current_date == birth_date :
    pywhatkit.sendwhatmsg(phone, f'Today is {name} birthday', send_hour, send_min)


# In[ ]:





# In[ ]:




