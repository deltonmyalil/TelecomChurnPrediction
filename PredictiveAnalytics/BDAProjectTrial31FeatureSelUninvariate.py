import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read into a pandas dataframe
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# view the first few records
data.head()

# check for data type and info
data.info()

# Preprocessing
# Converting Total Charges to a numerical data type.
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce')
#Let's check for nulls first
nulls = data.isnull().sum()
nulls[nulls > 0] # 11 null values in total charges
#in case of missing values, impute with 0
data.fillna(0,inplace=True)
#new feature - Internet(Yes- have internet service, No- do not have internet service)
#data['Internet'] = data['InternetService'].apply(lambda x : x if x=='No' else 'Yes') #redundant
#replace No phone service with No
data['MultipleLines'].replace('No phone service','No',inplace=True)
#train and target
y = data['Churn'].map({'Yes':1,'No':0})
X = data.drop(labels=['Churn','customerID'],axis=1).copy()
#find list of categorical columns for encoding
cat_cols = []
for column in X.columns:
    if column not in ['tenure','MonthlyCharges','TotalCharges']:
        cat_cols.append(column)
#Convert categorical columns to binary
# Alternative to OneHotEncoder
X= pd.get_dummies(X,columns=cat_cols)
X.info()

#feature selection using backward elimination
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.5*(1-0.5))
X = sel.fit_transform(X)
# HOLY MOTHER OF GOD, IT DROPPED EVERY CATEGORICAL VALUES
# AND STILL I AM GETTING GOOD ACCURACY WITH THE THREE FEATURES
# WHAT THE ACTUAL FUCK HAPPENED???

#Data partitioning
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Modeling

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logRegClassifier = LogisticRegression()
logRegClassifier.fit(X_train, y_train)
# classification using logRegressor
y_predLogReg = logRegClassifier.predict(X_test)
# Visualizing Logistic regression results
from sklearn.metrics import confusion_matrix
logRegConfMatrix = confusion_matrix(y_test, y_predLogReg) #1689/2113 = 79.933% accuracy; without new Internet field, 80.024%; with feature selection till p = 0.083, accuracy is 80.26%
#y_test_array = y_test.tolist()

out = pd.DataFrame(y)
out.info()
pd.value_counts(y)
#we get prior probability as [0.73, 0.27]

# Naive Bayes without using prior probability
from sklearn.naive_bayes import GaussianNB
nbClassifier = GaussianNB()
nbClassifier.fit(X_train, y_train)
# Classification using naive bayes
y_predNB = nbClassifier.predict(X_test)
# visualizing NB results
nbConfMatrix = confusion_matrix(y_test, y_predNB) # without prior 1446/2113 = 68.433% accuracy, without new Internet field, 69.85%; with FSS till p = 0.083, accuracy decreased to 69.096%

'''
nbClassifierWithPrior = GaussianNB(priors=[0.73, 0.27])
nbClassifierWithPrior.fit(X_train, y_train)
# Classification using naive bayes
y_predNBWithPrior = nbClassifierWithPrior.predict(X_test)
# visualizing NB results
nbWithPriorConfMatrix = confusion_matrix(y_test, y_predNBWithPrior) # without prior 1446/2113 = 68.433% accuracy
''' # With prior probabilities mentioned, I am getting same result

# KNN Classification
from sklearn.neighbors import KNeighborsClassifier
# considering 5 neighbors euclidian(minkowski with p=2) distance
KNNClassifier = KNeighborsClassifier(n_neighbors=20, p=2, metric='minkowski')
KNNClassifier.fit(X_train, y_train)
y_predKNN = KNNClassifier.predict(X_test)
# confusion matrix
knnConfMatrix = confusion_matrix(y_test, y_predKNN) # 78.513% accuracy; no change with new Internet field,  with FSS till p = 0.083, accuracy decresed to 75.911%
#tested with n_neighbours = 5, 10, 20, 40, 50

# Support Vector Machine Classification - taking too long - Skip for now
# This is raw, do with PCA
'''
from sklearn.svm import SVC
svcClassifier = SVC(kernel='linear', random_state=0, cache_size=7000) #increasing cache size prevented endless running
svcClassifier.fit(X_train, y_train)
y_predSVM = svcClassifier.predict(X_test)
# confusion matrix
svcConfMatrix = confusion_matrix(y_test, y_predSVM)
'''

# Using Principal Component Analysis
from sklearn.decomposition import PCA
pca  = PCA(n_components=3)
#pca = PCA(.80)
pca.fit(X_train)
#Transform X_test and X_train
X_train_PCA = pca.transform(X_train)
X_test_PCA = pca.transform(X_test)
# LogReg using PCSpace
pcaLogRegClassifier = LogisticRegression()
pcaLogRegClassifier.fit(X_train_PCA, y_train)
y_pred_logReg_PCA = pcaLogRegClassifier.predict(X_test_PCA)
# confusion matrix
pcaLogRegConfMatrix = confusion_matrix(y_test, y_pred_logReg_PCA) # for 3 PCi, Accuracy is 77.99%, with FSS, accuracy decreased to 77.851%


# The following is confirmed to work. But takes too damn long to learn.
# Uncomment in the final build with JNB
'''
# Using svm on principal component space
from sklearn.svm import SVC
svcPCAclassifier = SVC(kernel='linear', random_state=0)
svcPCAclassifier.fit(X_train_PCA, y_train)
y_pred_svc_PCA = svcPCAclassifier.predict(X_test_PCA)
#confusion matrix
pcaSVCConfMatrix = confusion_matrix(y_test, y_pred_svc_PCA) # for 3 Pci's, 77.94% accuracy
# PCA with 3 components worked faster for SVC.
# Direct SVC never stopped running.
# Hence did dimension reduction and used 3 PCi's
'''

# Using kernel SVM with rbf kernel

# Using decision tree classifier with information gain ie criterion='entropy
# Also visualize the dec tree created using pydot and graphviz. Code in my github.

'''
## Do this after each classification: Gives a detailed summary of your classification
from sklearn.metrics import classification_report
classification_report(y_test, y_pred)
'''
'''
import scikitplot as skplt
import matplotlib.pyplot as plt

y_true = # ground truth labels
y_probas = # predicted probabilities generated by sklearn classifier
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()

'''