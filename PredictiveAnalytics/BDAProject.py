import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read into a pandas dataframe
dataset = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# view the first few records
dataset.head()

# check for data type and info
dataset.info()

# to get the matrix of input variables
X = dataset.iloc[:, :].values # as the last column is out dependent var y
y = dataset.iloc[:, 20].values # the 20th column (from 0 to 20) is the depVar

# Data Preprocessing

#converting TotalCharges and MonthlyCharges to number
dataset.TotalCharges = pd.to_numeric(dataset.TotalCharges, errors="coerce")
dataset.MonthlyCharges = pd.to_numeric(dataset.MonthlyCharges, errors="coerce")
dataset.info()
# handling categorical Values
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
#for output variable
X[:, 20] = labelEncoder.fit_transform(X[:, 20])
df = pd.DataFrame(X)
y = X[:,20]
out = pd.DataFrame(y)
X = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
df = pd.DataFrame(X)

# to get the number of categorical values for index 1 ie gender
pd.value_counts(dataset.gender) # we get Male and Female ie 2 unique values
X[:, 1] = labelEncoder.fit_transform(X[:, 1])  # female became 0 and male became 1
#df = pd.DataFrame(X)

# to get the number of categorical values for index 3 ie partner
pd.value_counts(dataset.Partner) # we get No and Yes ie 2 unique values
X[:, 3] = labelEncoder.fit_transform(X[:, 3])  # yes became 1 and no became 0
#df = pd.DataFrame(X)

# to get the number of categorical values for index 4 ie Dependents
pd.value_counts(dataset.Dependents) # we get No and Yes ie 2 unique values
X[:, 4] = labelEncoder.fit_transform(X[:, 4])  # yes became 1 and no became 0
#df = pd.DataFrame(X)

# tenure ie index 5 is not categorical
pd.value_counts(dataset.tenure) # we get No and Yes ie 2 unique values

# to get the number of categorical values for index 6 ie PhoneService
pd.value_counts(dataset.PhoneService) # we get No and Yes ie 2 unique values
X[:, 6] = labelEncoder.fit_transform(X[:, 6])  # yes became 1 and no became 0
#df = pd.DataFrame(X)

# to get the number of categorical values for index 7 ie MultipleLines
pd.value_counts(dataset.MultipleLines) # we get No and Yes and No phone service ie 3 unique values
X[:, 7] = labelEncoder.fit_transform(X[:, 7])  # values got mapped to 0, 1, 2
#df = pd.DataFrame(X)

# to get the number of categorical values for index 8 ie InternetService
pd.value_counts(dataset.InternetService) # we get Fiber opptic, DSL and No
X[:, 8] = labelEncoder.fit_transform(X[:, 8])  # values got mapped to 0, 1, 2
#df = pd.DataFrame(X)

# to get the number of categorical values for index 9 ie OnlineSecurity
pd.value_counts(dataset.OnlineSecurity) # we get 3 unique values
X[:, 9] = labelEncoder.fit_transform(X[:, 9])  # values got mapped to 0, 1, 2
#df = pd.DataFrame(X)

# to get the number of categorical values for index 10 ie OnlineBackup
pd.value_counts(dataset.OnlineBackup) # we get 3 unique values
X[:, 10] = labelEncoder.fit_transform(X[:, 10])  # values got mapped to 0, 1, 2
#df = pd.DataFrame(X)

# to get the number of categorical values for index 11 ie DeviceProtection
pd.value_counts(dataset.DeviceProtection) # we get 3 unique values
X[:, 11] = labelEncoder.fit_transform(X[:, 11])  # values got mapped to 0, 1, 2
#df = pd.DataFrame(X)

# to get the number of categorical values for index 12 ie TechSupport
pd.value_counts(dataset.TechSupport) # we get 3 unique values
X[:, 12] = labelEncoder.fit_transform(X[:, 12])  # values got mapped to 0, 1, 2
#df = pd.DataFrame(X)

# to get the number of categorical values for index 12 ie StreamingTV
pd.value_counts(dataset.StreamingTV) # we get 3 unique values
X[:, 13] = labelEncoder.fit_transform(X[:, 13])  # values got mapped to 0, 1, 2
#df = pd.DataFrame(X)

# to get the number of categorical values for index 14 ie StreamingMovies
pd.value_counts(dataset.StreamingMovies) # we get 3 unique values
X[:, 14] = labelEncoder.fit_transform(X[:, 14])  # values got mapped to 0, 1, 2
#df = pd.DataFrame(X)

# to get the number of categorical values for index 15 ie Contract
pd.value_counts(dataset.Contract) # we get 3 unique values
X[:, 15] = labelEncoder.fit_transform(X[:, 15])  # values got mapped to 0, 1, 2
#df = pd.DataFrame(X)

# to get the number of categorical values for index 16 ie PaperlessBilling
pd.value_counts(dataset.PaperlessBilling) # we get 2 unique values
X[:, 16] = labelEncoder.fit_transform(X[:, 16])  # values got mapped to 0, 1
#df = pd.DataFrame(X)

# to get the number of categorical values for index 17 ie PaymentMethod
pd.value_counts(dataset.PaymentMethod) # we get 4 unique values
X[:, 17] = labelEncoder.fit_transform(X[:, 17])  # values got mapped to 0, 1

# now, customer id is a primary key, it is immaterial to the end result
pd.value_counts(dataset.customerID) # we get 7043 unique values
# encode it as serial number
X[:, 0] = labelEncoder.fit_transform(X[:, 0])  # values got mapped to 0, 1

# This is our final X matrix converted to a pandas dataframe for visualization in spyder
df = pd.DataFrame(X)
df.info()

# feature selection using backward elimination

#Data partitioning into training and test data:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Data preprocessing ends

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
