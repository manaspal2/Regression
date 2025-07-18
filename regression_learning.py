#################################################
# To find the relationship between one or more 
# independent variable and one dependent variable
#################################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
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

#######################################################################################################
# Mean Square error - One of the way to figure out the loss function
# Goal should be to keep the loss function minimum.
# In linear regression, large coefficinets like A1, B1(A1x1 + B1x2 .... = y) could cause over fitting
# Controlling the large co-efficient is names as regularization
# Of of the regularization is Ridge Regression where using hyperparameter alpha, the loss is minimized
# and control model complexity. 
# Zero value alpha -> Can cause overfitting
# Large alpha -> Can cause underfitting
# Find the effect of alpha in prediction
# With alpha value increase, we sould see the score will be reducing.
#######################################################################################################
scores = []
alpha_values = [0.1, 1.0, 10, 100, 100]
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train,y_train)
    y_pred = ridge.predict(X_test)
    scores.append(ridge.score(X_test,y_test))

print (scores)

#######################################################################################################
# Another method of reducing loss function is Lasso
# Lasso regression can be used to select the improtant features
# Lasso regression actually reduce the coefficient of not important features to zero or near about zero
# and increases the important features' coefficent.
#######################################################################################################
scores = []
Lasso_values = [0.1, 1.0, 10, 100, 100]
for lasso in Lasso_values:
    l1 = Lasso(alpha=lasso)
    l1.fit(X_train,y_train)
    y_pred = l1.predict(X_test)
    scores.append(l1.score(X_test,y_test))

print (scores)

#######################################################################################################
#Let's look at the coefficent adjustment
#######################################################################################################
X = diabetic_df.drop("Glucose", axis=1).values
y= diabetic_df["Glucose"].values
names = diabetic_df.drop("Glucose", axis=1).columns
l1 = Lasso(alpha=0.1)
l1_coeff = l1.fit(X,y).coef_
plt.bar(names,l1_coeff)
plt.xticks(rotation=45)
plt.show()
