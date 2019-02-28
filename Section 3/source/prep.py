"""
Prepare loan dataset to work with our classifier.

We're using a public domain dataset from:
https://www.kaggle.com/ninzaami/loan-predication
Code is heavily influenced by:
https://www.kaggle.com/hafidhfikri/loan-approval-prediction
Thanks!
"""
import pandas as pd

def get_data():
    """
    Get and clean up our loan dataset, return
    the whole dataset, X and Y.

    X - is a list of features (variables) that we
        have available (all of them has to be numerical)
    Y - is the result of loan application either 'N' for rejected,
        or 'Y' for approved
    """
    data = pd.read_csv("data/loan_approvals.csv")

    # Remove incomplete data (rows that has at least one
    # empty column).
    data.dropna(inplace=True)

    # Change the text values to unique numbers
    # We need to do that in our X
    data['Gender'] = data['Gender'].replace({'Female': 0, 'Male': 1})
    data['Married'] = data['Married'].replace({'No' : 0,'Yes' : 1})
    data['Dependents'] = data['Dependents'].replace({'0':0,'1':1,'2':2,'3+':3})
    data['Education'] = data['Education'].replace({'Not Graduate' : 0, 'Graduate' : 1})
    data['Self_Employed'] = data['Self_Employed'].replace({'No' : 0,'Yes' : 1})
    data['Property_Area'] = data['Property_Area'].replace({'Semiurban' : 0, 'Urban' : 1,'Rural' : 2})

    # Return the whole cleaned up dataset
    # X (without the Loan_ID column) and Y
    return data, data.iloc[:,1:12], data.iloc[:,12]

if __name__ == '__main__':
    data, X, Y=get_data()
    print('Raw XY:')
    print(data.head())
    print('X:')
    print(X.head())
    print(list(X.columns))
    print(list(X.iloc[0]))
    print('Y:')
    print(Y.head())
