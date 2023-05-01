#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


get_ipython().system('pip install seaborn')


# In[3]:


get_ipython().system('pip install plotly')


# In[4]:


df = pd.read_csv('Iris.csv')


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


# Create a scatter plot of sepal length versus sepal width
df.plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm')
plt.show()


# In[8]:


# Create a histogram of petal length
df['PetalLengthCm'].plot(kind='hist', bins=20)
plt.show()


# In[9]:


# Create a bar chart of the number of flowers in each species
species_count = df["Species"].value_counts()
plt.bar(species_count.index, species_count.values)
plt.xlabel("Species")
plt.ylabel("Number of Flowers")
plt.title("Number of Flowers per Species")

# Display the plot
plt.show()


# In[10]:


# Create a line plot of petal width over time (assuming the data is sorted by time)
plt.plot(df["PetalWidthCm"])
plt.xlabel("Time")
plt.ylabel("Petal Width (cm)")
plt.title("Petal Width over Time")

# Display the plot
plt.show()


# In[11]:


# Visualize with seaborn
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species')


# In[12]:


sns.set(style="ticks")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, orient="h", palette="Set2")
plt.title("Boxplot of Iris Dataset Features")
plt.show()


# In[13]:


sns.set(style="ticks")
sns.pairplot(df, hue="Species", height=2.5)
plt.show()


# In[15]:


# Visualize with plotly
fig = px.scatter(df, x='SepalLengthCm', y='SepalWidthCm', color='Species')
fig.show()


# In[18]:


sns.set(style="white")
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True)
plt.title("Correlation Heatmap of Iris Dataset Features")
plt.show()

