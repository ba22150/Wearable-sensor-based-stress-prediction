#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


# All required libraries are imported
import numpy as np
import pandas as pd
from scipy import stats
import glob as gb
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# # Loading and pre processing the Data 

# In[2]:


#Loading and pre processing the data
path = r"M:\Data Science(CE888)\Raw_data\**\*.csv"
filenames = gb.glob(path)
Particpant_num = 0
par_list = set()
df_BV = pd.DataFrame()
df_AC =  pd.DataFrame()
df_ED =  pd.DataFrame()
df_H =  pd.DataFrame()
for filename in filenames:
    if 'ACC.csv' in filename:
        Particpant_num = Particpant_num+ 1
        dp = pd.read_csv(filename, header = None,names = ['ACC_X', 'ACC_Y', 'ACC_Z'])
        st = dp.iloc[0,0]   #Starting time
        ft = dp.iloc[1,0]   #Frequency
        ft = int(ft)
        dp.drop(index =[0,1], inplace = True)
        dp = dp.reset_index()
        count = 0
        df1 = {}
        df_ACC = pd.DataFrame()
        while(count < len(dp)):
            df1= {
                'Participant' : Particpant_num,
                'Time Stamp' : st ,
                'ACC_X' : [np.mean(dp['ACC_X'][count : count+ft])], #Calculating the avg ACC X between the frequencies
                'ACC_Y' : [np.mean(dp['ACC_Y'][count : count+ft])], #Calculating the avg ACC Y between the frequencies
                'ACC_Z' : [np.mean(dp['ACC_Z'][count : count+ft])], #Calculating the avg ACC Z between the frequencies
                }
            st = st + 1
            count = count + ft
            df2 = pd.DataFrame(df1)
            df_ACC = pd.concat([df_ACC,df2], ignore_index=True)   #Making a dafa frame of values from each ACC
        df_AC = pd.concat([df_AC,df_ACC], ignore_index=True)      #Concatinating all data frames into 1
    elif 'BVP.csv' in filename:
        dp = pd.read_csv(filename, header = None, names = ["BVP"])
        st = dp.iloc[0,0]      #Starting time
        ft = dp.iloc[1,0]      #Frequency
        ft = int(ft)
        dp.drop(index =[0,1], inplace = True)
        dp = dp.reset_index()
        count = 0
        df1 = {}
        df_BVP = pd.DataFrame()
        while(count < len(dp)):
            df1= {
                'Participant' : Particpant_num,
                'Time Stamp' : st,                
                'BVP' : [np.mean(dp["BVP"][count : count+ft])],      #Calculating the avg BVP between the frequencies
                 }
            st = st + 1
            count = count + ft
            df2 = pd.DataFrame(df1)
            df_BVP = pd.concat([df_BVP,df2], ignore_index=True)     #Making a dafa frame of values from each BVP
        df_BV = pd.concat([df_BV,df_BVP], ignore_index=True)       #Concatinating all data frames into 1
    elif 'EDA.csv' in filename:
        dp = pd.read_csv(filename, header =None, names = ["EDA"])
        st = dp.iloc[0,0]     #Starting time
        ft = dp.iloc[1,0]     #Frequency
        ft = int(ft)
        dp.drop(index =[0,1], inplace = True)
        dp = dp.reset_index()
        count = 0
        df1 = {}
        df_EDA = pd.DataFrame()
        while(count < len(dp)):
            df1= {
                'Participant' : Particpant_num,
                'Time Stamp' : [st],
                'EDA' : [np.mean(dp["EDA"][count : count+ft])],    #Calculating the avg EDA between the frequencies
                }
            st = st + 1
            count = count + ft
            df2 = pd.DataFrame(df1)
            df_EDA = pd.concat([df_EDA,df2], ignore_index=True)    #Making a dafa frame of values from each EDA
        df_ED = pd.concat([df_ED,df_EDA], ignore_index=True)       #Concatinating all data frames into one
    elif 'HR.csv' in filename:
        dp = pd.read_csv(filename, header = None, names = ["HR"])
        st = dp.iloc[0,0]     #Starting time
        ft = dp.iloc[1,0]     #Frequency
        ft = int(ft)
        dp.drop(index =[0,1], inplace = True)
        dp = dp.reset_index()
        count = 0
        df1 = {}
        df_HR = pd.DataFrame()
        while(count < len(dp)):
            df1= {
                'Participant' : Particpant_num,
                'Time Stamp' : st,
                'HR' : [np.mean(dp["HR"][count : count+ft])],   #Calculating the avg HR between the frequencies
                 }
            st = st + 1
            count = count + ft
            df2 = pd.DataFrame(df1)
            df_HR = pd.concat([df_HR,df2], ignore_index=True)   #Making a dafa frame of values from each HR
        df_H = pd.concat([df_H,df_HR], ignore_index=True)      #Concatinating all data frames into one
    par_list.add(Particpant_num)


# In[3]:


#Merging all 4 final data frames into one
merge1 = pd.merge(df_AC,df_BV, on =['Time Stamp',"Participant"],how = 'outer')
merge2 = pd.merge(df_H,df_ED, on = ['Time Stamp',"Participant"],how = 'outer')
merged_df = pd.merge(merge1,merge2, on = ['Time Stamp',"Participant"],how = 'outer')


# In[4]:


merged_df.head()


# # Data Analysis

# In[5]:


#Number of missing values in the data
for columns in merged_df:
    print("Null values in",columns,"=",merged_df[columns].isnull().sum())


# In[6]:


#Dropping rows with missing values
merged_df.dropna(inplace = True)


# In[7]:


#HERE WE ARE MAKING A NEW COLUMN OUTPUT TO VALIDATE HOW MANY PARTICPANTS HAVE STRESS AND HOW MANY DON'T
#Adding Tags to the dataframe
file = []
# making list of data frames of tags w.r.t participants
for name in gb.glob(r'M:\Data Science(CE888)\Raw_data\*\tag*.csv'):  
    df = pd.read_csv(name,header= None)
    file.append(df)
count = 0
merged_df['Output'] = 0
for i in range(len(file)):
    count += 1
    for j in range(0,6):
        if j%2 == 0:
            cond = [np.logical_and(np.logical_and(merged_df["Time Stamp"]>file[i][0][j],merged_df["Time Stamp"]<file[i][0][j+1]),merged_df["Participant"]== count)]
            choice = [1]
            merged_df['Output'] = np.select(cond,choice,default = merged_df['Output'])


# In[8]:


merged_df.head()


# In[9]:


#Finding the data types
merged_df.dtypes


# In[10]:


# To verify the tags : 1-Stressed and 0- Not stressed
merged_df["Output"].value_counts() 


# In[11]:


#Dropping duplicates 
merged_df = merged_df.drop_duplicates() 


# In[12]:


# Merging the output to understand the correlation between the variables
merged_df.columns= ['Participant', 'Time Stamp', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'HR', 'EDA', 'Output']
corr = merged_df.corr()
corr['Output']


# In[13]:


merged_df


# # 1) Descriptive Statistics
# 

# 1. To view basic statistics of the dataframe:

# In[14]:


print(merged_df.describe())


# 2. To Compute the skewness and kurtosis of a specific column in the dataframe:

# In[15]:


#SKEWNESS AND KURTOSIS TO CHECK THE DISTRIBUTION OF THE DATA

skewness = merged_df['ACC_X'].skew()
print("Skewness of ACC_X,'ACC_X'",skewness)
skewness = merged_df['ACC_Y'].skew()
print("Skewness of ACC_Y,'ACC_Y'",skewness)
skewness = merged_df['ACC_Z'].skew()
print("Skewness of ACC_Z,'ACC_Z'",skewness)
skewness = merged_df['BVP'].skew()
print("Skewness of BVP,'BVP'",skewness)
skewness = merged_df['HR'].skew()
print("Skewness of HR,'HR'",skewness)
skewness = merged_df['EDA'].skew()
print("Skewness of EDA,'ACC_X'",skewness)
kurtosis = merged_df['ACC_X'].kurt()
kurtosis = merged_df['ACC_Y'].kurt()
kurtosis = merged_df['ACC_Z'].kurt()
kurtosis = merged_df['BVP'].kurt()
kurtosis = merged_df['HR'].kurt()
kurtosis = merged_df['EDA'].kurt()


# # 2) Data Visualization

# 1. To Compute the correlation between two columns in the dataframe:

# In[16]:


#correlation matrix
cormat = merged_df.corr()
#Plotting a heatmap
dataplot = sns.heatmap(cormat)


# 2. To compute a bar plot with the output column
# 

# In[17]:


# Bar plot for output
x= merged_df['Output'].value_counts()
print(x)
x.plot(kind='bar')

# Add labels and title
plt.xlabel('Offensive and Not Offensive tweets')
plt.ylabel('Tweets')
plt.title('Bar Plot for Offensive and Not Offensive tweets')

# Display the plot
plt.show()


# 3. To compute a seaborn scatter plot

# In[18]:


# Creating a pair plot  dataframe 'merged_df'
sns.pairplot(merged_df)


# 4. Time series plot

# In[19]:


# Convert the date column to datetime format
timeSeries_data = merged_df.copy()
timeSeries_data['Time Stamp'] = pd.to_datetime(timeSeries_data['Time Stamp'])

 # Create a new figure and axis object
fig, ax = plt.subplots()  

# Define the columns to plot
cols_to_plot = ['EDA']

# Plot the selected columns against the time axis
timeSeries_data[cols_to_plot].plot(ax=ax)

plt.rcParams["figure.figsize"] = [7, 3]
plt.rcParams["figure.autolayout"] = True

x = np.linspace(-15, 15, 100)
y = np.sin(x)

#Plot the time series data
plt.plot(x, y)

plt.xlim(-10, 10000)
plt.ylim(-10, 10)

# Add a title and labels to the x and y axes
plt.title('EDA over Time')
plt.xlabel('Time')
plt.ylabel('EDA')
# Show the plot
plt.show()


# In[20]:


# Convert the 'Time Stamp'  to a datetime format
timeSeries_data['Time Stamp'] = pd.to_datetime(timeSeries_data['Time Stamp'])
fig, ax = plt.subplots()
cols_to_plot = ['BVP']
timeSeries_data[cols_to_plot].plot(ax=ax)

plt.rcParams["figure.figsize"] = [7, 3]
plt.rcParams["figure.autolayout"] = True

x = np.linspace(-15, 15, 100)
y = np.sin(x)

plt.plot(x, y)

#Set the x and y limit of plot
plt.xlim(0, 900)
plt.ylim(-200, 200)

# Add a title and labels to the x and y axes
plt.title('BVP over Time')
plt.xlabel('Time')
plt.ylabel('BVP')
# Show the plot
plt.show()


# In[21]:


# Convert the 'Time Stamp' column to a datetime format
timeSeries_data['Time Stamp'] = pd.to_datetime(timeSeries_data['Time Stamp'])
fig, ax = plt.subplots()
cols_to_plot = ['HR']
timeSeries_data[cols_to_plot].plot(ax=ax)

# Setting the size and layout of the figure using the `rcParams` method of the `matplotlib.pyplot` module
plt.rcParams["figure.figsize"] = [7, 3]
plt.rcParams["figure.autolayout"] = True

x = np.linspace(-15, 15, 100)
y = np.sin(x)

plt.plot(x, y)

# Set the x and y limits of the plot
plt.xlim(0, 10000)
plt.ylim(-200, 200)

# Add a title and labels to the x and y axes
plt.title('HR over Time')
plt.xlabel('Time')
plt.ylabel('HR')
# Show the plot
plt.show()


# In[22]:


# Convert the 'Time Stamp' column to a datetime format
timeSeries_data['Time Stamp'] = pd.to_datetime(timeSeries_data['Time Stamp'])

# Create a new figure and axis object 
fig, ax = plt.subplots()
cols_to_plot = ['ACC_X','ACC_Y','ACC_Z']
timeSeries_data[cols_to_plot].plot(ax=ax)

plt.rcParams["figure.figsize"] = [7, 3]
plt.rcParams["figure.autolayout"] = True

x = np.linspace(-15, 15, 100)
y = np.sin(x)

plt.plot(x, y)

plt.xlim(0, 2000)
plt.ylim(-200, 200)

# Add a title and labels to the x and y axes
plt.title('ACC_X ACC_Y ACC_Z over Time')
plt.xlabel('Time')
plt.ylabel('ACC_X ACC_Y ACC_Z')
# Show the plot
plt.show()


# # Building Data Model

# In[23]:


# This function takes pandas DataFrame 'df' as input
def train_test(df):
    print(df.head())
    
    # Drop the 'Participant' column from the input dataframe 'df'
    df.drop('Participant', axis  = 1, inplace = True)
    
    # Creating a StandardScaler object 
    scaler = StandardScaler()
    x=df.drop(['Output'],axis=1)
    y=df['Output']
    
    # Splitting the data into training and testing sets using 'train_test_split' function from Scikit-learn library
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=13,shuffle = False)
    
    # Scaling the training and testing data 
    scaled_data_x_train = scaler.fit_transform(X_train)
    scaled_data_x_test = scaler.fit_transform(X_test)
    
    # Converting the scaled data arrays to pandas dataframes
    scaled_data_x_test = pd.DataFrame(scaled_data_x_test, columns = X_test.columns)
    scaled_data_x_train = pd.DataFrame(scaled_data_x_train, columns = X_train.columns)
    
     # Dropping the 'Time Stamp' from training and testing dataframes.
    scaled_data_x_train.drop('Time Stamp', axis  = 1, inplace = True)
    scaled_data_x_test.drop('Time Stamp', axis  = 1, inplace = True)
    
    
    #Calling the Random forest function
    score = random_forest(scaled_data_x_train,scaled_data_x_test,y_train,y_test)
    return score


# In[24]:


def random_forest(x_train,x_test,y_train,y_test):
    
    # RandomForestClassifier is initialized
    C2 = RandomForestClassifier(max_depth=10,random_state=10)
    C2 = C2.fit(x_train,y_train) 
    
    # predicts the labels of the test data
    Y_pred2 = C2.predict(x_test)
    score = classific_report(Y_pred2,y_test)
    return scores


# In[25]:


def classific_report(y_pred,y_test):
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    ##print("Precision:",metrics.precision_score(y_test, y_pred))
    #print("Recall:",metrics.recall_score(y_test, y_pred))
    #print("F1 score:",metrics.f1_score(y_test, y_pred))
    #Compute classification 
    classific_repo = classification_report(y_test,y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    print(classific_repo)
    print('Confusion Matrix', confusion_mat)
    
    #Display the confusion matrix as a plot
    cmd = ConfusionMatrixDisplay(confusion_matrix = confusion_mat, display_labels = [False, True])
    cmd.plot()
    plt.show()
    return metrics.accuracy_score(y_test, y_pred)


# In[26]:


#Below code extracts the data for the participant from the merged_df dataframe and trains a model.
scores =[]
for participant in par_list:
    
    df = merged_df.loc[merged_df['Participant'] == participant]
    score = train_test(df)
    scores.append(score)


    


# In[ ]:




