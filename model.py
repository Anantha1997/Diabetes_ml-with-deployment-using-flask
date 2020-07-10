import numpy as np
import pandas as pd
import os


# In[24]:


os.chdir("E:\\kaggle datas\\diabetes_ML")


# In[25]:


data = pd.read_csv("diabetes.csv")


# In[26]:


data.head()


# In[27]:


data.isnull().sum()


# In[28]:


dtype_df = data.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# In[29]:


data.groupby("Outcome").size()


# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt
colors = ["#0101DF", "#DF0101"]

sns.countplot('Outcome', data=data, palette=colors)
plt.title('diabetes Distributions \n (0: No diabetes || 1: diabetes)', fontsize=14)


# In[31]:


from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[32]:


features = "+".join(data.columns.difference(["Outcome"]))


# In[33]:


features


# In[34]:


#perform vif

a, b = dmatrices(formula_like= 'Outcome ~ ' + features,data=data,return_type="dataframe")
vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(b.values, i) for i in range(b.shape[1])]
vif["Features"] = b.columns


# In[35]:


vif


# In[36]:


X = data.drop('Outcome', axis = 1)
y = data.Outcome


# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =0)


# In[38]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# In[39]:


y_pred = logreg.predict(X_test)


# In[40]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[41]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print(cm)


# In[42]:


data.head()


# In[44]:


import pickle
# Saving model to disk
pickle.dump(logreg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# In[45]:


print(model.predict([[3, 100, 78, 25, 100, 48.9, 0.167, 80]]))


