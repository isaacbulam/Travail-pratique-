# -*- coding: utf-8 -*-
"""
Created on Sat May 11 02:18:13 2024

@author: Isaac BULA
"""

# Importer les librairies nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score

# Charger le jeu de données
url = "donneIris.csv"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = pd.read_csv(url, names=names)

#dataset = pd.read_csv('donneIris.csv')

# Résumer le jeu de données
print("Dimensions du jeu de données : ", dataset.shape)
print("\nAperçu des données:\n", dataset.head())
print("\nRésumé statistique de toutes les caractéristiques:\n", dataset.describe())
print("\nRépartition des données par rapport à la variable de classe:\n", dataset.groupby('class').size())

# Visualiser les données
dataset.hist()
plt.show()

# Diviser les données en features et target
X = dataset.iloc[ :, :-1].values
y = dataset.iloc[ :, -1].values

# Diviser les données en jeu d’entraînement et jeu de test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner les modèles
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


models = [('Logistic Regression', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('Decision Tree', DecisionTreeClassifier()),
          ('Support Vector Machine', SVC())]

best_model = None
best_accuracy = 0.0

for name, model in models :
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nModèle : ", name)
    print("Précision : ", accuracy)
    
    if accuracy > best_accuracy :
        best_accuracy = accuracy
        best_model = model

# Utiliser le meilleur modèle pour faire des prédictions sur le jeu de test
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nMeilleur modèle : ", best_model)
print("Précision du meilleur modèle : ", accuracy)
