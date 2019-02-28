"""
Use a trained GradientBoostingClassifier on
an example loan application to get the probablities
that will be either rejected or approved.
"""
from prep import get_data
from train import train
from pprint import pprint

if __name__ == '__main__':
    dfields=['Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    test_X=[[0.0, 1.0, 2.0, 1.0, 1.0, 3000.0, 0.0, 50000.0, 360.0, 1.0, 1.0]]
    _, X, Y = get_data()
    gbc, _=train(X, Y)
    # You have to fit the model with the data
    # before you can use it for predictions.
    gbc.fit(X, Y)
    pprint(list(zip(dfields, test_X[0])))
    r=gbc.predict(test_X)
    print('Would this person get the loan?:', r[0])
    rp=gbc.predict_proba(test_X)
    print('Probablity of loan being approved: %d%%, rejected: %d%%' % (rp[0][1]*100, rp[0][0]*100))
