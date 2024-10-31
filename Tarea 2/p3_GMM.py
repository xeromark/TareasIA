import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Cargar el dataset
df = pd.read_csv('dataset.csv')

# seleccion del  Subconjunto
x = df[['BMI']]
y = df['HighBP']

# Entrenar el modelo con GMM
gmm = GaussianMixture(n_components=2, random_state=42) 
gmm.fit(x)

# Predecir las etiquetas
predictions = gmm.predict(x)

# Añadir las predicciones al DataFrame
df['GMM_Predicted'] = predictions

# Calcular el número de clusters
n_clusters = len(np.unique(predictions))
print(f'Número de clusters: {n_clusters}')

# Visualizar resultados
plt.figure(figsize=(10, 6))

sns.scatterplot(x='BMI', y='HighBP', data=df, hue='GMM_Predicted', palette='viridis', alpha=0.6)

plt.title('Clasificación GMM de BMI vs HighBP')
plt.xlabel('BMI')
plt.ylabel('High Blood Pressure (HighBP)')
plt.legend(title='GMM Clusters')
plt.grid()
plt.show()

# Silhouette Score
silhouette_avg = silhouette_score(x, predictions)
print(f'Silhouette Score: {silhouette_avg}')

means = gmm.means_  # Medias de las gaussianas
covariances = gmm.covariances_  # Covarianzas de las gaussianas

print("Medias de las gaussianas:")
print(means)

print("\nCovarianzas de las gaussianas:")
print(covariances)
