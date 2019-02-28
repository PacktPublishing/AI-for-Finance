"""
Train and test our GradientBoostingClassifier.

Note:
We don't split our dataset into train and test set
since we're using cross validation (cross_val_score)
that will randomly split our dataset, train our model
on each split and we will get the list of accuracies.

In production we often do both - do the train/test split
AND do the cross validation on the training dataset.
"""
from prep import get_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def train(X, Y):
  gbc = GradientBoostingClassifier()
  scores = cross_val_score(gbc, X, Y,cv=5)
  return gbc, scores

if __name__ == '__main__':
  _, X, Y = get_data()
  gbc, scores=train(X, Y)
  print('Loan approval accuracy for this model is %.2f%%' %(scores.mean()*100))
