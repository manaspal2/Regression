################################################################
# Logistic regression is another type of linear regression model
# This model predicts the probablity of a data to be part of a 
# binary class.
# Generally if p > 0.5, it is considered as 1
# Generally if p < 0.5, it is considered as 0
################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# Read data from CSV file
df = pd.read_csv("train_telecom_data.csv")
print (df.head())
###############################################################################################
# In this following example a single feature variable is chosen to predict a target variable
###############################################################################################
# Create the target and feature variables
X = df[["total_day_charge", "total_eve_charge"]].values
y= df["churn"].values
print (type(y))
y = np.where(y == 'yes', 1, 0)
#print (y)

logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state=42)

logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

#Predict probability
y_pred_probs = logreg.predict_proba(X_test)[:,1]
print (y_pred_probs)

fpr, tpr, threashold = roc_curve(y_test, y_pred_probs, pos_label=1)
#print (fpr)
#print (tpr)
#print (threashold)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr,tpr)
plt.xlabel('False poistive rate')
plt.ylabel('True positive rate')
plt.title('Logistic regression curve')
plt.show()

# Area under the ROC curve is the way to proceed to measure the score of the Ligistic regression.
print (roc_auc_score(y_test, y_pred_probs))