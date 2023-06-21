# ML Classificação

* Curso Alura: Machine Learning: Classificação por trás dos panos

- Variável Dummie
- Balancear os dados
- Modelo K-nearest neighbors (KNN)
- Padronizar os dados
- Distância Euclidiana
- Teorema de Naive Bayes
- Modelo Bernoulli Naive Bayes
- Árvore de decisão
- Matriz de confusão
- Acurácia
- Precisão
- Recall

    - import pandas as pd
    - import seaborn as sns
    - from imblearn.over_sampling import SMOTE
    - from sklearn.preprocessing import StandardScaler
    - import numpy as np
    - from sklearn.model_selection import train_test_split
    - from sklearn.neighbors import KNeighborsClassifier
    - from sklearn.naive_bayes import BernoulliNB
    - from sklearn.tree import DecisionTreeClassifier
    - from sklearn.metrics import confusion_matrix
    - from sklearn.metrics import accuracy_score
    - from sklearn.metrics import precision_score
    - from sklearn.metrics import recall_score