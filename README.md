# Predict_Credit_Card_Fraud_Using_Regression

Predict Credit Card Fraud
Credit card fraud is one of the leading causes of identify theft around the world. In 2018 alone, over $24 billion were stolen through fraudulent credit card transactions. 
Financial institutions employ a wide variety of different techniques to prevent fraud, one of the most common being Logistic Regression.

In this project, you are a Data Scientist working for a credit card company. You have access to a dataset (based on a synthetic financial dataset), 
that represents a typical set of credit card transactions. transactions.csv is the original dataset containing 200k transactions. 
For starters, we’re going to be working with a small portion of this dataset, transactions_modified.csv, which contains one thousand transactions. 
Your task is to use Logistic Regression and create a predictive model to determine if a transaction is fraudulent or not.




# Load the Data
1.
The file transactions_modified.csv contains data on 1000 simulated credit card transactions. Let’s begin by loading the data into a pandas DataFrame named transactions. Take a peek at the dataset using .head() and you can use .info() to examine how many rows are there and what datatypes the are. How many transactions are fraudulent? Print your answer.


# Clean the Data
2.
Looking at the dataset, combined with our knowledge of credit card transactions in general, we can see that there are a few interesting columns to look at. We know that the amount of a given transaction is going to be important. Calculate summary statistics for this column. What does the distribution look like?


3.
We have a lot of information about the type of transaction we are looking at. Let’s create a new column called isPayment that assigns a 1 when type is “PAYMENT” or “DEBIT”, and a 0 otherwise.



4.
Similarly, create a column called isMovement, which will capture if money moved out of the origin account. This column will have a value of 1 when type is either “CASH_OUT” or “TRANSFER”, and a 0 otherwise.



5.
With financial fraud, another key factor to investigate would be the difference in value between the origin and destination account. Our theory, in this case, being that destination accounts with a significantly different value could be suspect of fraud. Let’s create a column called accountDiff with the absolute difference of the oldbalanceOrg and oldbalanceDest columns.


# Select and Split the Data

6.
Before we can start training our model, we need to define our features and label columns. Our label column in this dataset is the isFraud field. Create a variable called features which will be an array consisting of the following fields:

amount
isPayment
isMovement
accountDiff
Also create a variable called label with the column isFraud.


7.
Split the data into training and test sets using sklearn‘s train_test_split() method. We’ll use the training set to train the model and the test set to evaluate the model. Use a test_size value of 0.3.



# Normalize the Data

8.
Since sklearn‘s Logistic Regression implementation uses Regularization, we need to scale our feature data. Create a StandardScaler object, .fit_transform() it on the training features, and .transform() the test features.



# Create and Evaluate the Model

9.
Create a LogisticRegression model with sklearn and .fit() it on the training data.

Fitting the model find the best coefficients for our selected features so it can more accurately predict our label. We will start with the default threshold of 0.5.



10.
Run the model’s .score() method on the training data and print the training score.

Scoring the model on the training data will process the training data through the trained model and will predict which transactions are fraudulent. The score returned is the percentage of correct classifications, or the accuracy.



11.
Run the model’s .score() method on the test data and print the test score.

Scoring the model on the test data will process the test data through the trained model and will predict which transactions are fraudulent. The score returned is the percentage of correct classifications, or the accuracy, and will be an indicator for the sucess of your model.

How did your model perform?


12.
Print the coefficients for our model to see how important each feature column was for prediction. Which feature was most important? Least important?



# Predict With the Model

13.
Let’s use our model to process more transactions that have gone through our systems. There are three numpy arrays pre-loaded in the workspace with information on new sample transactions under “New transaction data”

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])
Create a fourth array, your_transaction, and add any transaction information you’d like. Make sure to enter all values as floats with a .!


14.
Combine the new transactions and your_transaction into a single numpy array called sample_transactions.


15.
Since our Logistic Regression model was trained on scaled feature data, we must also scale the feature data we are making predictions on. Using the StandardScaler object created earlier, apply its .transform() method to sample_transactions and save the result to sample_transactions.



16.
Which transactions are fraudulent? Use your model’s .predict() method on sample_transactions and print the result to find out.

Want to see the probabilities that led to these predictions? Call your model’s .predict_proba() method on sample_transactions and print the result. 
The 1st column is the probability of a transaction not being fraudulent, and the 2nd column is the probability of a transaction being fraudulent 
(which was calculated by our model to make the final classification decision).
