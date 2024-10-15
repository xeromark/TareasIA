import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score

# Cargar el dataset
df = pd.read_csv('dataset.csv') # https://www.kaggle.com/datasets/kandij/diabetes-dataset



# Dividir el dataset
X = df[["BMI"]]  # Solo usamos la característica BMI
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Entrenar el modelo KNN
k_values = [1, 5, 7, 11, 21]
results = {}

accuracies = []
recalls = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    # Calcular métricas
    accuracies.append(accuracy_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))

# Mostrar resultados
print(accuracies)
print(recalls)


# Graficar los resultados
plt.figure(figsize=(10, 5))

plt.plot(k_values, accuracies, label='Accuracy', marker='o', color='b', linestyle='--')
plt.plot(k_values, recalls, label='Recall', marker='s', color='r', linestyle='-.')

plt.xlabel('Número de vecinos (K)')
plt.ylabel('Valor de la métrica')
plt.title('Rendimiento del Modelo KNN para diferentes valores de K')
plt.legend(loc='best')
plt.grid(True)
plt.show()