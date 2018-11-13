%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea

dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataset.info()

dataset.head()

list(dataset.columns.values)

# Check for missing
dataset.isnull().sum()

# Check for unique vals
dataset.nunique()

# Separating the columns
id_col     = ['customerID']
target_col = ['Churn']
category_cols = [col for col in dataset.columns if dataset[col].nunique() <= 4 and col != 'Churn']
numeric_cols = [col for col in dataset.columns if col not in category_cols and col != 'Churn' and col != 'customerID']
# Categorical
print(category_cols)

# Numeric
print(numeric_cols)

plt.rc("font", size=14)
sea.set(style="white") #white background style for seaborn plots
sea.set(style="whitegrid", color_codes=True)
fig , axes = plt.subplots(nrows = 6 ,ncols = 3,figsize = (15,30))
for i, item in enumerate(category_cols):
    if i < 3:
        ax = dataset[item].value_counts().plot(kind ='bar', ax=axes[0, i], rot = 0)
    elif i >=3 and i < 6:
        ax = dataset[item].value_counts().plot(kind ='bar', ax=axes[1, i - 3], rot = 0)
    elif i >= 6 and i < 9:
        ax = dataset[item].value_counts().plot(kind ='bar', ax=axes[2, i - 6], rot = 0)
    elif i < 12:
        ax = dataset[item].value_counts().plot(kind ='bar', ax=axes[3, i - 9], rot = 0)
    elif i < 15:
        ax = dataset[item].value_counts().plot(kind ='bar', ax=axes[4, i - 12], rot = 0)
    elif i < 18:
        ax = dataset[item].value_counts().plot(kind ='bar', ax=axes[5, i - 15], rot = 0)
    ax.set_title(item)
	
# Plotting churners and non churners
sea.countplot(x='Churn', data=dataset);

# Categorical data as a relation to churn
fig , axes = plt.subplots(nrows = 6 ,ncols = 3,figsize = (20,50))
for i, col in enumerate(category_cols):
    if i < 3:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[0,i],rot = 0)
        ax = sea.countplot(x=col, hue = dataset['Churn'], data=dataset, ax=axes[0, i])
    elif i >=3 and i < 6:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[1,i-3],rot = 0)
        ax = sea.countplot(x=col, hue = dataset['Churn'], data=dataset, ax=axes[1, i - 3])
    elif i >= 6 and i < 9:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[2,i-6],rot = 0)
        ax = sea.countplot(x=col, hue = dataset['Churn'], data=dataset, ax=axes[2, i - 6])
    elif i < 12:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[3,i-9],rot = 0)
        ax = sea.countplot(x=col, hue = dataset['Churn'], data=dataset, ax=axes[3, i - 9])
    elif i < 15:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[4,i-12],rot = 0)
        ax = sea.countplot(x=col, hue = dataset['Churn'], data=dataset, ax=axes[4, i - 12])
    elif i < 18:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[5,i-15],rot = 0)
        ax = sea.countplot(x=col, hue = dataset['Churn'], data=dataset, ax=axes[5, i - 15])
    ax.set_title(col)

# imputing TotalCharges
dataset['TotalCharges'] = dataset["TotalCharges"].replace(" ", np.nan)
dataset["TotalCharges"] = dataset["TotalCharges"].fillna(0)
dataset["TotalCharges"] = dataset["TotalCharges"].astype(float)
	
# Correlation Matrix
corr_matrix = dataset[['MonthlyCharges', 'TotalCharges'  , 'tenure']].corr()
plt.figure(figsize=(15, 15))
corrmap = sea.heatmap(corr_matrix, square=True, annot=True)

