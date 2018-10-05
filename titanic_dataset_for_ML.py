#importing all libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

#importing dataset
titanic_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

#checking dimensions of data
titanic_data.shape

#Printing first 10 and last 10 rows of the data
titanic_data[0:10].append(titanic_data[-10:])

# Printing all the attribute/column names
titanic_data.columns

# Printing datatypes of all attributes
titanic_data.dtypes

#printing description of data
titanic_data.describe()

#let's start exploratory data analysis
##checking the distribution of Age attribute
titanic_data.boxplot(column=['Age'], return_type='axes');
plt.ylabel("Number")

titanic_data.boxplot(column=['SibSp'], return_type='axes');
plt.ylabel("Number")

titanic_data.boxplot(column=['Fare'], return_type='axes');

#checking the relation of attributes against our class label
gender_df = titanic_data.pivot_table(index = "Sex", values = "Survived")
gender_df.plot.bar()
plt.show()

plt.scatter(titanic_data["Age"], titanic_data["Survived"])
plt.xlabel("Age")
plt.ylabel("Survived")
plt.show()

parch_df = titanic_data.pivot_table(index = "Parch", values = "Survived")
parch_df.plot.bar()
plt.show()

plt.scatter(titanic_data["Fare"], titanic_data["Survived"])
plt.xlabel("Fare")
plt.ylabel("Survived")
plt.show()

sibsp_df = titanic_data.pivot_table(index = "SibSp", values = "Survived")
sibsp_df.plot.bar()
plt.xlabel("SibSp")
plt.ylabel("Survived")
plt.show()

# let's start Preprocessing of this dataset
#I am copying the dataset
df = titanic_data.copy()

#dropping duplicates if any
df.drop_duplicates(inplace=True)

#checking for missing values
df.isna().sum()

#looking at the data
df.Embarked.value_counts()

#filling missing values in 'Embarked' attribute with most frequent value
freq_port = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(freq_port)

#dealing with missing values in 'Age' attribute
for name_string in df['Name']:
    df['Title']=df['Name'].str.extract('([A-Za-z]+)\.',expand=True)
    
    

#replacing the rare title with more common one.
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
df.replace({'Title': mapping}, inplace=True)

df['Title'].value_counts()

titles=['Mr','Miss','Mrs','Master','Rev','Dr']
for title in titles:
    age_to_impute = df.groupby('Title')['Age'].median()[titles.index(title)]
    df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = age_to_impute
    
#we are not going to use 'Cabin' attribute for the prediction, therefore not doing anything for it
    
#transforming nominal attribute into numeric
df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

#let's choose attributes for our model

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
#I am ommiting the features Name and Ticket number, which I believe have nothing to do with survival chances.
#In the wreck women, children and elite had higher chances of survival, therefore I have chosen features 
#accordingly.  

#separating class attribute from rest of the data
X_train = df.drop('Survived', axis = 1)
y_train = df['Survived']
X_test = test_data.copy()


### Using sklearn train to use a simple decision Tree Classifier on training data

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#checking accuracy of decision tree
dt_training_outcome = classifier.predict(X_train)
print('Training accuracy...', accuracy_score(y_train, dt_training_outcome))

### Train a simple Logistic Regression Model on training data
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)

lr_training_outcome = logisticRegr.predict(X_train)
print('Training accuracy...', accuracy_score(y_train, lr_training_outcome))

#Will continue later
    