import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print ("############################################")
print ("### Step 1: Loading the insurance dataset ##")
print ("############################################")
insurance_data_path = 'insurance.csv'
insurance = pd.read_csv(insurance_data_path)
print (insurance.head())

print ("##############################################")
print ("#### Step 2: Loading the validation dataset ##")
print ("##############################################")
validation_data_path = 'validation_dataset.csv'
validation = pd.read_csv(validation_data_path)
print (validation.head())

def dataPreprocess(df):
    print ("######################################################")
    print ("## Step 3: Check how many missing values in dataset ##")
    print ("######################################################")
    print (df.isna().sum())

    print ("#####################################################")
    print ("## Step 4: After dropping all the NaN values       ##")
    print ("#####################################################")
    columnList = df.columns.to_list()
    print (columnList)
    df = df.dropna(subset=columnList)
    #print (df.head())

    print ("#########################################################")
    print ("## Step 5: After dropping NaN, how many missing values ##")
    print ("#########################################################")
    print (df.isna().sum())

    print ("################################################")
    print ("## Step 6: Check unique values in each column ##")
    print ("################################################")
    for elem in columnList:
        print (df[elem].unique())
    
    print ("###########################################################")
    print ("## Step 7: After cleaning up $ symbol from charge column ##")
    print ("###########################################################")
    if 'charges' in columnList:
        df.loc[:, 'charges'] = df['charges'].replace(to_replace="\$([0-9,\.]+).*", value=r"\1", regex=True)

    print ("################################################################")
    print ("## Step 8: Check unique values in each column after $ removal ##")
    print ("################################################################")
    for elem in columnList:
        print (df[elem].unique())

    print ("##################################################")
    print ("## Step 9: Convert all the values to small case ##")
    print ("##################################################")
    for elem in columnList:
        print (elem, "data type is", df[elem].dtypes)
        if df[elem].dtypes == "object":
            print (elem, "column value is converted to small case")
            df.loc[:, elem] = df[elem].astype(str).str.lower()

    print ("#######################################################")
    print ("## Step 10: Check unique values in each column after ##") 
    print ("## converting to smaller case                        ##")
    print ("#######################################################")
    for elem in columnList:
        print (df[elem].unique())

    print ("##################################################")
    print ("## Step 11: After dropping the negative values  ##")
    print ("##################################################")
    columns = ['age', 'children']
    for elem in columns:
        df = df[df[elem] >= 0]

    for elem in columnList:
        print (df[elem].unique())

    print ("#####################################################")
    print ("## Step 12: Replacing m and f with male and female ##")
    print ("#####################################################")
    df.loc[:, 'sex']  = df['sex'].str.replace('^man', 'male', regex=True)
    df.loc[:, 'sex']  = df['sex'].str.replace('^woman', 'female', regex=True)
    df.loc[:, 'sex']  = df['sex'].str.replace('^m$', 'male', regex=True)
    df.loc[:, 'sex']  = df['sex'].str.replace('^f$', 'female', regex=True)
    for elem in columnList:
        print (df[elem].unique())

    print ("####################################################################")
    print ("## Step 13: Convert the categorial variable to numerical variable ##")
    print ("####################################################################")
    df = pd.get_dummies(df, 
                columns=["sex", "smoker", "region"], 
                dtype=int)
    columnList = df.columns.to_list()
    df['age'] = df['age'].astype(int)
    df['children'] = df['children'].astype(int)

    return df

preproc_df = dataPreprocess(insurance)
print (preproc_df.head())

print ("########################")
print ("#### Target Variables ##")
print ("########################")
y = preproc_df[['charges']]
print (y.head())

print ("#########################")
print ("#### Feature Variables ##")
print ("#########################")
X = preproc_df.drop(columns=["charges"])
print (X.head())

print ("###############################")
print ("## Split Train and Test data ##")
print ("###############################")
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size=0.4,
                                                    random_state=42,
                                                    )

print ("###################")
print ("## Training data ##")
print ("###################")
print (X_train.head())
print (y_train.head())

print ("###################")
print ("## Test data     ##")
print ("###################")
print (X_test.head())
print (y_test.head())

print ("#######################################")
print ("## Build the Linear Regression Model ##")
print ("#######################################")
reg = LinearRegression()
sc = StandardScaler(with_mean=True)

X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.fit_transform(X_test)

reg.fit(X_train_sc,y_train)
y_pred = reg.predict(X_test_sc)

print ("########################")
print ("## Predicted y values ##")
print ("########################")
print (y_pred)

print ("#########################")
print ("### Original y values ###")
print ("#########################")
print (y_test)

if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test)
if not isinstance(y_test, pd.DataFrame):
    y_test = pd.DataFrame(y_test)
if not isinstance(y_pred, pd.DataFrame):
    y_pred = pd.DataFrame(y_pred, columns=['predict'])


test_df = pd.concat([X_test, y_test], axis=1)
print (test_df.shape)
test_df.reset_index(drop=True, inplace=True)
test_pred_df = pd.concat([test_df, y_pred], axis=1)
print (test_pred_df.head())
scores = cross_val_score(reg, 
                         X, 
                         y, 
                         scoring='r2', 
                         cv=5)
print (scores)
r2_score = np.mean(scores)
print ("R Square value : ", r2_score)

print ("#######################################")
print ("## Test Linear Regression Model      ##")
print ("#######################################")
preproc_df = dataPreprocess(validation)
print (preproc_df.head())
predicted_ch = reg.predict(preproc_df)
validation["predicted_charges"] = predicted_ch
validation.loc[validation['predicted_charges'] < 0, 'predicted_charges'] = 1000
print (validation.head())