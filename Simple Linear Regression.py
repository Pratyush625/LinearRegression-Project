
# coding: utf-8

# # Simple Linear Regression

# In this example we will consider sales based on 'TV' marketing budget.
# 
# In this nootbook, we'll build a linear regression model to predict 'Sales' using 'TV' as the predictor variable'.

# # Understanding the Data

# Below are the following steps
# 
# 1.Importing data using the pandas library
# 
# 2.Understanding the structure of the data

# In[1]:


import pandas as pd


# In[4]:


advertising=pd.read_csv('tvmarketing.csv')


# Now, let's check the structure of the advertising dataset

# In[5]:


advertising


# In[6]:


# To display the first 5 rows
advertising.head()


# In[7]:


# To display the last 5 rows
advertising.tail()


# In[8]:


# Let's check the columns 
advertising.info()


# In[9]:


# Let's check the shape of the DataFrame (rows & columns )
advertising.describe()


# # Visualising Data Using Seaborn

# In[10]:


# Conventional way to import seaborn 
import seaborn as sns

# To visualise in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


# Visualise the relationship between the features and the response using scatterplots
sns.pairplot(advertising, x_vars='TV', y_vars='Sales', size=7, aspect=0.7, kind= 'scatter')


# # Performing Simple Linear Regression

# Equation of linear regression
# 
# y = c + m1x1+m2x2+....+mnxn
# 
# . y is the response (dependent variable)
# . c is the intercept
# . m1 is the coefficient for the first feature (slope)
# . mn is the coefficient for the nth feature
# 
# In our case:
# 
# y = c +m1 * TV
# 
# The m value are called the model coeffiecients or model parameters.

# # Generic Steps in Model Building using Sklearn

# # Preparing x and y

# . The sklearn library expects x(feature variable) and y(target variable) to be NumPy arrays.
# 
# . However, x can be determined as Pandas is built over NumPy.

# In[52]:


# Putting feature variable to x  # Using iloc function to segregate the feature & target
x = advertising.iloc[:,:-1].values
x


# In[53]:


# Putting target variable to y
y = advertising.iloc[:,1].values
y


# # Splitting Data into Training and Testing Sets

# In[55]:


# random_state is the seed used by the random number generator, it can be any integer.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=100)


# In[56]:


print(type(x_train))
print(type(x_test))
print(type(y_train))
print(type(y_test))


# In[59]:


#check the shape
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test.shape)


# # Performing Linear Regression

# In[60]:


# import LinearRegresion from sklearn
from sklearn.linear_model import LinearRegression

# Representing LinearRegression as linreg(creating LinearRegression object)
linreg = LinearRegression()

#Fit the model using linreg.fit()
linreg.fit(x_train,y_train)


# # Coefficients Calculation

# In[61]:


# Print the intercept and the coefficients
print(linreg.intercept_)
print(linreg.coef_)


# y=6.989 + 0.0464 * TV
# 
# Now let's use the this equation to predict our sales

# In[62]:


# Making the prediction on the testing dataset
y_pred = linreg.predict(x_test)
y_pred


# In[64]:


y_test


# In[63]:


type(y_pred)


# # Computing RMSE and R^2 values

# In[66]:


# Find out r^2 score
from sklearn.metrics import r2_score


# In[76]:


r2_score(y_test,y_pred)


# It means the accuracy of the model is around 60%

# In[77]:


# Find out RMSE
from sklearn.metrics import mean_squared_error


# In[79]:


Mean_square_error=mean_squared_error(y_test,y_pred)
Mean_square_error


# It means the model is not able to match around 8% values only

# # Graphical representation

# In[83]:


#plotting Regression line in training set
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,linreg.predict(x_train),color='blue')
plt.title('TV vs SALES (Training set)')
plt.xlabel('TV')
plt.ylabel('SALES')
plt.show()


# In[87]:


#plotting Regression line in test set
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,linreg.predict(x_train), color='blue')
plt.title('TV vs Sales (Testing set)')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()


# In[85]:


# Actual vs predict
plt.scatter(y_test,y_pred)
plt.xlabel('y test')
plt.ylabel('y_pred')
plt.plot()

