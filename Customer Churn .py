#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# In[2]:


# Load and explore the data (EDA)
file1 = "churn-bigml-20.csv"
file2 = "churn-bigml-80.csv"
data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)
data = pd.concat([data1, data2], ignore_index=True)


# In[3]:


# Display and check data
print(data.head())
print(data.info())


# In[4]:


# Transform categorical variables into numerical using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['State', 'International plan', 'Voice mail plan'], drop_first=True)


# In[5]:


# Select features (X) and target (y)
X = data_encoded.drop('Churn', axis=1)
y = data_encoded['Churn']


# In[6]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Features scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[17]:


# Logistic Regression
model = LogisticRegression()


# In[22]:


model.fit(X_train, y_train)


# In[15]:


# Prediction on the test set
y_pred = model.predict(X_test)


# In[18]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))


# In[20]:


# Data visualization
import matplotlib.pyplot as plt
churn_counts = data['Churn'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(churn_counts.index, churn_counts.values, color=['blue', 'red'])
plt.title('Customer Churn Counts')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.xticks([False, True], ['Not Churn', 'Churn'])
plt.show()

