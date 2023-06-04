import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import codecademylib3

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
print(transactions.info())

# How many fraudulent transactions? = 282
fraud = transactions.isFraud[transactions.isFraud == 1]
print(fraud)

# Summary statistics on amount column
print(transactions.amount.describe())

# Create isPayment field
transactions.isPayment = transactions.type == 0
transactions.isPayment[transactions.type.isin(["PAYMENT","DEBIT"])] = 1

# Create isMovement field
transactions.isMovement = transactions.type == 0
transactions.isMovement[transactions.type.isin(["CASH_OUT","TRANSFER"])] = 1

# Create accountDiff field
transactions.accountDiff = transactions.oldbalanceOrg - transactions.oldbalanceDest

# Create features and label variables
features = transactions[["amount","isPayment","isMovement","accountDiff"]]
label = transactions.isFraud
print(features.head())

# Split dataset
x_train, x_test, y_train, y_test =train_test_split(features, label, train_size=0.7, test_size=0.3)

# Normalize the features variables
scaler = StandardScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)

# Fit the model to the training data
model = LogisticRegression()
model.fit(x_train, y_train)

# Score the model on the training data = 0.4142857142857143
print(model.score(x_train, y_train))

# Score the model on the test data = 0.39666666666666667
print(model.score(x_test, y_test))

# Print the model coefficients 5.05630062e-07 -> Amount most import predictor

print(model.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])


# Create a new transaction
your_transaction = np.array([10000.31, 1.0, 0.0, 500.5])

# Combine new transactions into a single array
sample_transactions = np.stack((transaction1,transaction2,transaction3,your_transaction))

# Normalize the new transactions + Predict fraud on the new transactions + Predict fraud on the new transactions

sample_transactions= scaler.transform(sample_transactions)
print(model.predict(sample_transactions))
print(model.predict_proba(sample_transactions))