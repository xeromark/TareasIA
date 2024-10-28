from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn import metrics

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('dataset.csv') # https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

# Seleccionar la característica del dataset de BMI y HighBP
x = df[['BMI']]    # Indice de masa corporal (BMI)
y = df['HighBP']  # si tiene o no Hipertensión (HighBP)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)
# el 50% de los datos se usan para pruebas y el resto para entrenar (parametro asociado a test_size)

model = LogisticRegression(max_iter=100000) # Crear el modelo de regresión logística
model.fit(X_train, y_train) # entrenar el modelo

# Predecir en el conjunto de prueba
predictions = model.predict(X_test)

print("Luego se tiene el siguiente gráfico:")

# Graficar los datos
plt.scatter(x, y, color='blue', label='Datos', alpha=0.5)

# Graficar la frontera de decisión
# Crear un rango de valores de BMI
BMI_range = np.linspace(df['BMI'].min()-1, df['BMI'].max()+1, 300).reshape(-1, 1)

# Calcular z = b0 + b1 * BMI para cada valor en el rango
z = model.intercept_ + model.coef_ * BMI_range

# Aplicar la función logística G(z) = 1 / (1 + e^(-z))
G_z = 1 / (1 + np.exp(-z))

# Graficar la curva de regresión logística
plt.plot(BMI_range, G_z, color='red', label='Frontera de decisión')


# Configuración del gráfico
plt.title('Clasificación de Hipertensión basada en el BMI')
plt.xlabel('Índice de Masa Corporal (BMI)')
plt.ylabel('Resultado de Hipertensión (0 = No, 1 = Sí)')
plt.yticks([0, 1])
plt.legend()
plt.grid()
plt.show()


# Para mostrar el gráfico de la matriz de confusión
cm = confusion_matrix(y_test, predictions)
cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels = [False, True])
cm_display.plot(cmap='plasma')
plt.title('Logistic Regression')
plt.show()

print("Donde:")

# Se extraen TP, FP, FN y TN
TN, FP, FN, TP = cm.ravel()  # Desempaquetar los valores
print(f'TP (Verdaderos Positivos): {TP}')
print(f'FP (Falsos Positivos): {FP}')
print(f'FN (Falsos Negativos): {FN}')
print(f'TN (Verdaderos Negativos): {TN} \n')

print("A partir de esto, se obtienen las siguiente métricas:")


# Se calculan las métricas
accuracy = accuracy_score(y_test, predictions)
print(f'Acuraccy: {accuracy:.2f}')

recall = recall_score(y_test, predictions)
print(f'Recall: {recall:.2f}\n')
