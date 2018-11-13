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
# Building optimal model using backward elimination

import statsmodels.formula.api as sm
# Append an array of 1s to the 0th col of the matrix for BackElim to work
# the reason is y = b0 + b1x1 + b2x2 + ..., b0 needs a variable as well, else it will be discarded I guess
X = np.append(arr= np.ones((7043,1)).astype(int), values = X, axis=1)
# Now preprocessing for backwardElim is complete
X_opt = X
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p value is for col 42
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p value is for col 14, so delete that
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p value is for col 20, so delete that
X_opt = X_opt[:,[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p s for col 32, delete it
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 28, delete it
#X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 35, delete it
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 37, delete it
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 38, delete it
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 28, delete it
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 19, delete it
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 5, delete it
X_opt = X_opt[:,[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 32, delete it. Highest p was .452
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 11, delete it. Highest p was .178
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 10, delete it. Highest p was .083
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 29, delete it. Highest p was .019
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 9, delete it. Highest p was .013
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 8, delete it. Highest p was .075
#X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]
X_opt = X_opt[:,[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  # OrdinaryLeastSquares
regressor_OLS.summary() # Highest p is for col 8, delete it. Highest p was .075



# I think I will stop dropping stuff from here
# Do the following line with caution. Only if you want Feature selection to be involved
X = X_opt

# Feature Scaling using Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


'''# Feature Scaling using Normalization
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
X = normalizer.fit_transform(X)
'''

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
from sklearn.metrics import roc_curve, auc
logRegConfMatrix = confusion_matrix(y_test, y_predLogReg) #1689/2113 = 79.933% accuracy; without new Internet field, 80.024%; with feature selection till p = 0.083, accuracy is 80.26%; till p = 0.011, accuracy is 80.265

fpr, tpr, thresholds = roc_curve(y_predLogReg, y_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='Logistic Regression (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#y_test_array = y_test.tolist()
''' #use this for a detailed report of the calssification
from sklearn.metrics import classification_report
classification_report(y_test, y_predLogReg)
'''
'''
out = pd.DataFrame(y)
out.info()
pd.value_counts(y)
#we get prior probability as [0.73, 0.27]
'''

# Naive Bayes without using prior probability
from sklearn.naive_bayes import GaussianNB
nbClassifier = GaussianNB()
nbClassifier.fit(X_train, y_train)
# Classification using naive bayes
y_predNB = nbClassifier.predict(X_test)
# visualizing NB results
nbConfMatrix = confusion_matrix(y_test, y_predNB) # without prior 1446/2113 = 68.433% accuracy, without new Internet field, 69.85%; with FSS till p = 0.083, accuracy decreased to 69.096%, with p = 0.11, accuracy decreased to 67.628
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_predNB, y_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='Naive Bayes (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

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
knnConfMatrix = confusion_matrix(y_test, y_predKNN) # 78.513% accuracy; no change with new Internet field,  with FSS till p = 0.083, accuracy decresed to 75.911%, with p = 0.11, accuracy increased to 76.194
# ROC
fpr, tpr, thresholds = roc_curve(y_predKNN, y_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='KNN (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#tested with n_neighbours = 5, 10, 20, 40, 50

# Support Vector Machine Classification - taking too long - Skip for now
# This is raw, do with PCA

from sklearn.svm import SVC
svcClassifier = SVC(kernel='linear', random_state=0, cache_size=7000) #increasing cache size prevented endless running
svcClassifier.fit(X_train, y_train)
y_predSVM = svcClassifier.predict(X_test)
# confusion matrix
svcConfMatrix = confusion_matrix(y_test, y_predSVM)


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
pcaLogRegConfMatrix = confusion_matrix(y_test, y_pred_logReg_PCA) # for 3 PCi, Accuracy is 77.99%, with FSS, accuracy decreased to 77.851%, (with FSS As expected)
fpr, tpr, thresholds = roc_curve(y_pred_logReg_PCA, y_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='PCA Logistic Regression (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# The following is confirmed to work. But takes too damn long to learn.
# Uncomment in the final build with JNB
'''
# Using svm on principal component space
from sklearn.svm import SVC
svcPCAclassifier = SVC(kernel='linear', random_state=0)
svcPCAclassifier.fit(X_train_PCA, y_train)
y_pred_svc_PCA = svcPCAclassifier.predict(X_test_PCA)
#confusion matrix
pcaSVCConfMatrix = confusion_matrix(y_test, y_pred_svc_PCA) # for 3 Pci's, 76.620% accuracy
fpr, tpr, thresholds = roc_curve(y_pred_svc_PCA, y_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='SVM PCA (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# PCA with 3 components worked faster for SVC.
# Direct SVC never stopped running.
# Hence did dimension reduction and used 3 PCi's
'''
# Using SVM


# Using kernel SVM with rbf kernel
# This also works, but takes a long time. Uncomment in final build
'''
from sklearn.svm import SVC
kernelSVCclassifier = SVC(kernel="rbf", degree=3, probability=True, random_state=0) #enables probability estimates, 
kernelSVCclassifier.fit(X_train, y_train)
y_predKernelSVM = kernelSVCclassifier.predict(X_test)
# confusion matrix
kernelSVMconfusionMatrix = confusion_matrix(y_test, y_predKernelSVM) # Accuracy with FSS = 76.620%
# ROC
fpr, tpr, thresholds = roc_curve(y_predKernelSVM, y_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='Kernel SVM (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

'''

# Using decision tree classifier with information gain ie criterion='entropy
from sklearn.tree import DecisionTreeClassifier
decisionTreeClassifierEntropy = DecisionTreeClassifier()
decisionTreeClassifierEntropy.fit(X_train, y_train)
y_predDecTreeEntropy = decisionTreeClassifierEntropy.predict(X_test)
# confusion matrix
decTreeEntropyConfusionMatrix = confusion_matrix(y_test, y_predDecTreeEntropy) # Accuracy = 72.787%
# ROC
fpr, tpr, thresholds = roc_curve(y_predDecTreeEntropy, y_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='Decision Tree (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Visualizing the decision tree
# Dont even try te commented lines, it will crash the PC when the image opens
# It works though
'''
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dotData = StringIO()
export_graphviz(decisionTreeClassifierEntropy, out_file=dotData, filled=True, rounded=True, special_characters=True)
decTreeEntropyGraph = pydotplus.graph_from_dot_data(dotData.getvalue())
Image(decTreeEntropyGraph.create_png())
'''
# save the png file generated and then compress it online.

# Using random forest classifier
from sklearn.ensemble import RandomForestClassifier
randomForestClassifier = RandomForestClassifier(n_estimators=10, criterion="entropy", max_features=15, random_state=0)
randomForestClassifier.fit(X_train, y_train)
y_predRandomForest = randomForestClassifier.predict(X_test)
# confusion Matrix
randomForestConfMatrix = confusion_matrix(y_test, y_predRandomForest) # 78.7032% accuracy
# ROC
fpr, tpr, thresholds = roc_curve(y_predRandomForest, y_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='Random Forest (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Light Gradient Boosting Method
from lightgbm import LGBMClassifier
lgbmClassifier = LGBMClassifier(learning_rate=0.1, objective="binary", random_state=0, max_depth=12)
lgbmClassifier.fit(X_train, y_train)
y_predLGBM = lgbmClassifier.predict(X_test)
# confuction matrix
lgbmConfMatrix = confusion_matrix(y_test, y_predLGBM) # 79.7444% accuracy
# ROC
fpr, tpr, thresholds = roc_curve(y_predLGBM, y_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='Light Gradient Boosting (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# XGboost
from xgboost import XGBClassifier
xgbClassifier = XGBClassifier(learning_rate=0.1, max_depth=4)
#xgbClassifier = XGBClassifier(learning_rate=0.9, max_depth=4, random_state=0)
xgbClassifier.fit(X_train, y_train)
y_predXGB = xgbClassifier.predict(X_test)
# confusion matrix
xgbConfMatrix = confusion_matrix(y_test, y_predXGB) # 80.785 % accuracy
# ROC
fpr, tpr, thresholds = roc_curve(y_predXGB, y_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='XG Boost (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# MLP
from sklearn.neural_network import MLPClassifier
#final build, verbose = True
nnClassifier = MLPClassifier(activation="relu", solver="sgd", random_state=0, max_iter=85) 
nnClassifier.fit(X_train, y_train)
y_predMLP = nnClassifier.predict(X_test)
# confusion matrix
mlpConfMatrix = confusion_matrix(y_test, y_predMLP) #Standardized, with default accuracy = 79.9386%
# Roc for mlp
fpr, tpr, thresholds = roc_curve(y_predMLP, y_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='MLP Classifier (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# ROC PLOT - Use this to plot ROC.
# Plotting all for comparison
fpr, tpr, thresholds = roc_curve(y_predLogReg, y_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='Logistic Regression (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--') # Straight
fpr, tpr, thresholds = roc_curve(y_predNB, y_test)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', lw=1, label='Naive Bayes (area = %0.2f)' % roc_auc)
fpr, tpr, thresholds = roc_curve(y_predKNN, y_test)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='green', lw=1, label='KNN (area = %0.2f)' % roc_auc)
fpr, tpr, thresholds = roc_curve(y_pred_logReg_PCA, y_test)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='cyan', lw=1, label='LogReg PCA (area = %0.2f)' % roc_auc)
'''
fpr, tpr, thresholds = roc_curve(y_pred_svc_PCA, y_test)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='magenta', lw=1, label='SVC PCA (area = %0.2f)' % roc_auc)
'''
'''
fpr, tpr, thresholds = roc_curve(y_predKernelSVM, y_test)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='red', lw=1, label='Kernel SVM (area = %0.2f)' % roc_auc)
'''
fpr, tpr, thresholds = roc_curve(y_predDecTreeEntropy, y_test)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='yellow', lw=1, label='Decision Tree (area = %0.2f)' % roc_auc)
fpr, tpr, thresholds = roc_curve(y_predRandomForest, y_test)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='black', lw=1, label='Random Forest (area = %0.2f)' % roc_auc)
fpr, tpr, thresholds = roc_curve(y_predLGBM, y_test)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='violet', lw=1, label='Light Gradient Boosting (area = %0.2f)' % roc_auc)
fpr, tpr, thresholds = roc_curve(y_predXGB, y_test)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkgoldenrod', lw=1, label='XG Boosting (area = %0.2f)' % roc_auc)
fpr, tpr, thresholds = roc_curve(y_predMLP, y_test)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='chocolate', lw=1, label='MLP (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


'''
import scikitplot as skplt
import matplotlib.pyplot as plt

y_true = # ground truth labels
y_probas = # predicted probabilities generated by sklearn classifier
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()
'''
#import tensorflow
