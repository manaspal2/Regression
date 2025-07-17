#################################################
# To find the relationship between one or more 
# independent variable and one dependent variable
#################################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

# Read data from CSV file
diabetic_df = pd.read_csv("diabetes.csv")
print (diabetic_df.head())
###############################################################################################
# In this following example a single feature variable is chosen to predict a target variable
###############################################################################################
# Create the target and feature variables
X = diabetic_df.drop("Glucose", axis=1).values
y= diabetic_df["Glucose"].values
#print (X)
#print (type(X))
#print (y)
#print (type(y))

# Making prediction from a single feature
X_bmi = X[:,4]
print (X_bmi)
print (X_bmi.shape, y.shape)

X_bmi = X_bmi.reshape(-1,1)
print (X_bmi.shape, y.shape)

# Plot the dependent variables against independent variables
plt.scatter(X_bmi, y)
plt.ylabel("Glucose Level")
plt.xlabel("Bmi")
plt.show()

# Create the linear regression model, fit the data in that model and predict
reg = LinearRegression()
reg.fit(X_bmi, y)
prediction = reg.predict(X_bmi)
plt.scatter(X_bmi, y, color="blue")
plt.plot(X_bmi, prediction, color="red")
plt.ylabel("Glucose Level")
plt.xlabel("Bmi")
plt.show()

############################################
# y = ax + b
# Goal a and b are called model coefficient
# To find a and b, define a error finction and goal is to reduce the error minimal
# This error function is called loss/cost function
# Ordinary least square = RSS = This is the sum of squared distance between fit line and original data point
############################################

###############################################################################################
# In this following example a multiple feature variables are chosen to predict a target variable
###############################################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state=42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

###########################################################################################
# R Squared which quantifies the variance of the target variable explained by the features
# A higher R-squared generally suggests a better fit of the model to the data, meaning the 
# independent variables account for a larger portion of the variability in the dependent variable. 
# Finding the R squared value is below;
###########################################################################################
print (reg_all.score(X_test, y_test))

###########################################################################################
# Root mean square error - another parameter to find the model performance
###########################################################################################
print (root_mean_squared_error(y_test, y_pred))

###########################################################################################
# R-square value is dependent on how to split the data
# To avoid the bias of data split, there is a solution : Cross-Validation 
###########################################################################################
kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()
cv_result = cross_val_score(reg, X, y, cv=kf)
print (cv_result)
