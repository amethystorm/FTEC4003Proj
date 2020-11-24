# FTEC4003 Course Project Task 2: How many Customers Stay
## 1. Background
- This data comes from clients of a bank. These clients have already had accounts in this bank. Nowadays, the bank wants to model whether they will stay or not in the future. The task is to do the binary classification based on the given information, which gives extra information to the bank to stabilize the customers.
## 2. Data Set Information
- The data are attributes of customers' basic information. 
- train.csv
  - The training set with 13 input attributes and 1 output attribute (i.e. class attribute)
- assignment-test.csv
  - The testing set with 13 input attributes. You need to identify the class of each item. 
#### This data set contains two files:
1. train.csv
	- The training set with known labels
2. assignment-test.csv
	- The testing set without labels (the "Exited" Attribute).

#### Other files
1. samplesubmission.csv:
	
- This is a sample file to show the output format. Wrong format will lead to unknown result.
	
2. evaluate_2.macOS:
	- This is a command line tool to evaluate your result. We will use F1-measure to measure your result. 
	
  - Usage: Press "command + space" to open spotlight search and type in "terminal", then type in the following command in the terminal. You should replace
	```./submission_2.csv``` with your own path to the submission_2.csv.  Please note that ```./```denote the current position of the command line and ```submission_2.csv``` denote your submission file name.
```bash
./evaluate_2.macOS ./submission_2.csv
```

3. evaluate_2.linux:
	- Usage: Press "ctrl + alt + t" to launch a terminal and input the following command.
	- Other  notification details are as introduced in the "macOS setting".
```bash
./evaluate_2.linux ./submission_2.csv
```

4. evaluate_2.exe:
  - Usage: Press "command + r" and then type in "cmd" in the dialog box to launch a terminal. Then type in the command:
  - Other notification details are as introduced in the "macOS setting".
```bash
./evaluate_2.exe ./submission_2.csv
```

## 3. Goal

- The classification goal is to predict if the customer will leave this bank and choose other competitors in the future (i.e, Identify the value of feature 'Exited', 1 for yes and 0 otherwise).

## 4. Attribute Information
#### a) Input variables

**customers' basic information**

- RowNumber: the number of rows
- CustomerId: the id of the customer in this bank.
- Surname: the surname of the customer
- CreditScore: personal credit score for an account.
- Geography: the location of the customer.
- Gender: the gender of the customer.
- Age: the age of the customer.
- Tenure: the valid time of the account.
- Balance: the amount of money in the account.
- NumOfProducts: the number of products the customer buys.
- HasCrCard: The number of Credit Card the customer owns.
- IsActiveMember: whether active in the recent period.
- EstimatedSalary: the estimated salary of the custome

#### b) Output variable

- Exited: whether this customer will leave in the future.

