# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:59:59 2024

@author: USER
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("sonar.all-data", names=["X1", "X2", "X3", "X4", "X5", "X60", "Y"])

print(data.shape)
print(data.dtypes)
print(data.head())
print(data.describe())
print(data['Y'].value_counts())

data.hist(figsize=(10, 10))
plt.subplots_adjust(bottom=0.1)
plt.show()

data.plot(kind="scatter", x="X1", y="Y", alpha=0.5)
plt.show()


X = data.drop("Y", axis=1)
y = data["Y"]

X_train, X_test, X_train, X_test= train_test_split(X, y, test_size=0.2, random_state=42)

cv= KFold(n_splits=10, shuffle=True, random_state=42)

def evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{model.__class__.__name__}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print()
    
models = [
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(),
    SVC()
    ]

for model in models:
    evaluate_model()
    
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for model in models:
    pipe = Pipeline([
        ("scaler", scaler),
        ("model", model)
        ])
    pipe.fit(X_train_scaled, y_train)
    y_pred = pipe.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    racall = racall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{model.__class__.__name} (standardisé):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print()
    


    