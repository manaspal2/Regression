#################################################
# To find the relationship between one or more 
# independent variable and one dependent variable
#################################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

diabetic_df = pd.read_csv("diabetes.csv")
print (diabetic_df.head())

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