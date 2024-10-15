from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos
df = pd.read_csv('dataset.csv') # https://www.kaggle.com/datasets/kandij/diabetes-dataset


# Seleccionar la característica BMI y el objetivo Outcome
x = df[['BMI']]
y = df['Outcome']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)
# el 50% de los datos se usaría para entrenar

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
predictions = model.predict(X_test)
#print(X_test)
# También puedes calcular la precisión del modelo
accuracy = model.score(X_test, y_test)
print(f'Precisión del modelo: {accuracy:.2f}')

recall = recall_score(y_test, predictions)
print(f'Precisión del modelo: {recall:.2f}')

print("Donde:")

cm = confusion_matrix(y_test, predictions)
# Extraer TP, FP, FN y TN
TN, FP, FN, TP = cm.ravel()  # Desempaquetar los valores
print(f'TP (Verdaderos Positivos): {TP}')
print(f'FP (Falsos Positivos): {FP}')
print(f'FN (Falsos Negativos): {FN}')
print(f'TN (Verdaderos Negativos): {TN}')


# Graficar los datos
plt.scatter(x, y, color='blue', label='Datos', alpha=0.5)

# Graficar la frontera de decisión
# Crear un rango de valores de BMI
BMI_range = np.linspace(df['BMI'].min()-1, df['BMI'].max()+1, 300).reshape(-1, 1)
# Calcular las probabilidades de pertenecer a la clase positiva
BMI_range_df = pd.DataFrame(BMI_range, columns=['BMI'])

probabilities = model.predict_proba(BMI_range_df)[:, 1]

# Graficar la frontera de decisión
plt.plot(BMI_range, probabilities, color='red', label='Frontera de decisión')

# Configuración del gráfico
plt.title('Clasificación de Diabetes basada en el BMI')
plt.xlabel('Índice de Masa Corporal (BMI)')
plt.ylabel('Resultado de Diabetes (0 = No, 1 = Sí)')
plt.yticks([0, 1])
plt.legend()
plt.grid()
plt.show()