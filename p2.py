import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn import metrics

# Cargar el dataset
df = pd.read_csv('dataset.csv') # https://www.kaggle.com/datasets/kandij/diabetes-dataset

# Dividir el dataset
X = df[["BMI"]]  # Solo usamos la característica BMI
y = df["HighBP"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Entrenar el modelo KNN
k_values = [1, 5, 7, 11, 21]
results = {}

accuracies = []
recalls = []
cms = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    # Se extraen TP, FP, FN y TN
    cms.append(confusion_matrix(y_test, y_pred))

    # Calcular métricas
    accuracies.append(accuracy_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))

# Mostrar resultados
print("A partir de esto, se obtienen las siguiente métricas para cada k:")
print("accuracies =",accuracies)
print("recalls =",recalls)

print("\n Luego se obtiene el siguiente gráfico:\n")

# Graficar los resultados
plt.plot(k_values, accuracies, label='Accuracy', marker='o', color='b', linestyle='--')
plt.plot(k_values, recalls, label='Recall', marker='s', color='r', linestyle='-.')


plt.xlabel('Número de vecinos (K)')
plt.ylabel('Valor de la métrica')
plt.title('Rendimiento del Modelo KNN para diferentes valores de K')
plt.legend(loc='best')
plt.grid(True)
plt.show()

print("\n Matriz de confusión de el resultado con mejor Recall \n")

# Para mostrar el gráfico de la matriz de confusión
cm_display = metrics.ConfusionMatrixDisplay(cms[2], display_labels = [False, True])
cm_display.plot(cmap='plasma')
plt.title('Logistic Regression')
plt.show()

print("\n Donde se tiene que:")
# Se extraen TP, FP, FN y TN
TN, FP, FN, TP = cms[2].ravel()  # Desempaquetar los valores
print(f'TP (Verdaderos Positivos): {TP}')
print(f'FP (Falsos Positivos): {FP}')
print(f'FN (Falsos Negativos): {FN}')
print(f'TN (Verdaderos Negativos): {TN} \n')

