#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hvplot.pandas
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px


# In[2]:


# Read the dataframe

df = pd.read_csv(r'C:\Users\WANCHI\Desktop\CV\materials\Course 8 case study\E-Commerce Riviews\Dataset E-Commerce Reviews.csv', 
                 index_col=0)
df.head()


# In[3]:


df.info()


# In[4]:


df.drop_duplicates()


# In[5]:


# Check nan Values

df.isnull().sum()


# In[6]:


# Drop null values

df.dropna(subset=['Title', 'Review Text'], inplace=True)
df.isnull().sum()


# In[7]:


df[['Rating']].describe()


# In[8]:


# Plot the histogram of rating

fig1 = px.histogram(df, x="Rating")
fig1.update_traces(marker_color="chocolate",marker_line_color='rgb(48,23,8)',
                  marker_line_width=1)
fig1.update_layout(title_text='Product Rating')
fig1.show()


# In[9]:


# Plot the horizontal bar graph of class name with rating = 1,2

df_class = df[(df['Rating'] == 1) | (df['Rating'] == 2)]
df_class['Class Name'].value_counts().hvplot.barh()


# In[10]:


# Plot the horizontal bar graph of clothing ID with rating = 1,2

df_id = df[(df['Rating'] == 1) | (df['Rating'] == 2)]
df_id['Clothing ID'].value_counts()[:10].hvplot.barh()


# In[11]:


# Create smaller data frame with data columns in need

df.drop(labels=['Clothing ID','Age', 'Recommended IND', 'Positive Feedback Count', 'Division Name', 
                'Department Name','Class Name' ], axis=1, inplace=True)
df.head()


# In[12]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud


# In[13]:


# Create stopwords list:

stopwords = set(stopwords.words('english'))
textt = " ".join(review for review in df['Review Text'])
wordcloud = WordCloud(stopwords=stopwords).generate(textt)

# plot the WordCloud image 
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[14]:


# Assign reviews with rating > 3 as positive sentiment
# rating < 3 negative sentiment
# remove rating = 3
df = df[df['Rating'] != 3]
df['sentiment'] = df['Rating'].apply(lambda rating : +1 if rating > 3 else -1)
df.head(10)


# In[15]:


# Split df - positive and negative sentiment:
positive = df[df['sentiment'] == 1]
negative = df[df['sentiment'] == -1]


# In[16]:


## beautiful, cute and great removed because they were included in negative sentiment

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
stopwords.update(['beautiful','cute','great']) 


# In[17]:


# Wordcloud — positive sentiment

pos = " ".join(review for review in positive.Title)
wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()


# In[18]:


# Wordcloud — negative sentiment

neg = " ".join(review for review in negative.Title)
wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud3)
plt.axis("off")
plt.show()


# In[19]:


# Order the data with sentiment

df.sort_values(by=['sentiment'], inplace=False, ascending=True ).head(30)


# In[20]:


# Count the number of negative reviews

df[df['sentiment'] == -1].count()


# In[21]:


# Filter 'Title' contains 'wanted,' 'love' or 'like'

df1 = df[df['Title'].str.contains('Wanted|wanted|Love|love|Like|like')]
df1[df1['sentiment'] == -1].head(30)


# In[22]:


# Count the number of 'Title' which contains 'wanted,' 'love' or 'like'

df1[df1['sentiment'] == -1].count()


# In[23]:


# Calculate the proportion of 'Title' which contains 'wanted,' 'love' or 'like'

df1[df1['sentiment'] == -1].count() / df[df['sentiment'] == -1].count() *100


# In[24]:


# Filter 'Title' contains 'disappointed' 
df2 = df[df['Title'].str.contains( 'disappointed')]
df2[df2['sentiment'] == -1].head(30)


# In[25]:


# Calculate the proportion of 'Title' which contains 'disappointed' 

df2[df2['sentiment'] == -1].count() / df[df['sentiment'] == -1].count() *100


# In[26]:


# Filter 'Title' contains 'fit,' 'big,' or 'huge'
df3 = df[df['Title'].str.contains( 'Fit|fit|Big|big|Huge|huge')]
df4 = df3[df3['Title'].str.contains( 'Huge disappointment|Big disappointment') == False]
df4[df4['sentiment'] == -1]


# In[27]:


# Calculate the proportion of 'Title' which contains 'fit,' 'big,' or 'huge'

df4[df4['sentiment'] == -1].count() / df[df['sentiment'] == -1].count() *100


# In[28]:


# Filter 'Title' contains 'quality' or 'fabric'
df5 = df[df['Title'].str.contains( 'Quality|quality|Fabric|fabric')]
df5[df5['sentiment'] == -1]


# In[29]:


# Calculate the proportion of 'Title' which contains 'quality' or 'fabric'

df5[df5['sentiment'] == -1].count() / df[df['sentiment'] == -1].count() *100


# In[30]:


# Filter 'Title' contains 'color' 
df6 = df[df['Title'].str.contains('Color|color')]
df7 = df6[df6['Title'].str.contains('Great color|Lovely color|Cute color|Nice color|Pretty color|Beautiful color')== False] 
df7[df7['sentiment'] == -1]


# In[ ]:





# In[31]:


# Distribution of reviews with sentiment

df['sentimentt'] = df['sentiment'].replace({-1 : 'negative'})
df['sentimentt'] = df['sentimentt'].replace({1 : 'positive'})
fig2 = px.histogram(df, x="sentimentt")
fig2.update_traces(marker_color="lightcoral",marker_line_color='rgb(48,23,8)',
                  marker_line_width=1)
fig2.update_layout(title_text='Product Sentiment')
fig2.show()


# In[32]:


# Data Cleaning

def remove_punctuation(text):
    final = "".join(u for u in text if u not in (",", "?", ".", ";", ":", "!",'"',"'"))
    return final
df['Title'] = df['Title'].apply(remove_punctuation)


# In[33]:


dfNew = df[['Title','sentiment']]
dfNew.head()


# In[34]:


# Split data into training and testing data

index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <= 0.8]
test = df[df['random_number'] > 0.8]


# In[35]:


# Count vectorizer:

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Title'])
test_matrix = vectorizer.transform(test['Title'])


# In[36]:


# Logistic regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[37]:


X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']


# In[38]:


# Fit model on data

lr.fit(X_train,y_train)


# In[39]:


predictions = lr.predict(X_test)
predictions[0:20]


# In[40]:


# Find accuracy, precision, recall

from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
confusion_matrix(predictions,y_test)


# In[41]:


print(classification_report(predictions,y_test))

