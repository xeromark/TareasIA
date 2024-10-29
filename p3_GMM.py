import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 1. Cargar el dataset
data = pd.read_csv('dataset.csv')  # Reemplaza con la ruta a tu dataset


# 2. Seleccionar las columnas relevantes
X = data[['BMI']]
y = data['HighBP']

# Convertir a formato adecuado para GMM
X = X.values.reshape(-1, 1)

# 3. Entrenar el modelo GMM
gmm = GaussianMixture(n_components=2, random_state=42)  # Ajustar n_components según sea necesario
gmm.fit(X)

# Predecir las etiquetas
predictions = gmm.predict(X)

# Añadir las predicciones al DataFrame
data['GMM_Predicted'] = predictions

# 4. Visualizar los resultados
plt.figure(figsize=(10, 6))

# Gráfico de dispersión de BMI y HighBP
sns.scatterplot(x='BMI', y='HighBP', data=data, hue='GMM_Predicted', palette='viridis', alpha=0.6)

# Añadir títulos y etiquetas
plt.title('Clasificación GMM de BMI vs HighBP')
plt.xlabel('BMI')
plt.ylabel('High Blood Pressure (HighBP)')
plt.legend(title='GMM Clusters')
plt.grid()
plt.show()

# 5. Evaluación de Resultados
# Silhouette Score
silhouette_avg = silhouette_score(X, predictions)
print(f'Silhouette Score: {silhouette_avg}')

# Adjusted Rand Index
ari = adjusted_rand_score(y, predictions)
print(f'Adjusted Rand Index: {ari}')
