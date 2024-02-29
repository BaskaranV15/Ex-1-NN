<H3>ENTER YOUR NAME:BASKARAN V</H3>
<H3>ENTER YOUR REGISTER NO:212222230020</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("/content/Churn_Modelling.csv")
df

df.isnull().sum()
df.info()

correlation=df.corr()
plt.figure(figsize=(17,7))
sns.heatmap(correlation,annot=True,cmap='coolwarm')
plt.show()

dropvalue=['RowNumber','Surname','CustomerId']
df.drop(dropvalue,axis=1,inplace=True)
df['Geography']=df['Geography'].astype('category')
df['Gender']=df['Gender'].astype('category')
df['Geography']=df['Geography'].cat.codes
df['Gender']=df['Gender'].cat.codes

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

scaler=MinMaxScaler()
df1=scaler.fit_transform(df)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
```


## OUTPUT:

![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/118703522/5e130900-afbe-41ab-ac7c-125ff17076dc)
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/118703522/32f5ae0f-e735-4d7c-9c4a-bb74623ce7d1)
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/118703522/df62d3a2-d9b0-4af6-9fdf-8ab3524356cc)
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/118703522/b4911336-43ed-45ee-be29-de00954e9941)
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/118703522/a4e75bb7-8078-49cb-9f70-be1d0fb59d71)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
