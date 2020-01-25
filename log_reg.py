import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=43)

lr = LogisticRegression()
lr.fit(X_train, y_train)

pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print(f'Accuracy:',accuracy)
print(f'Confusion Matrix:\n',confusion_matrix(y_test, pred))
