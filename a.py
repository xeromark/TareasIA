# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Cargar el dataset
data = pd.read_csv('dataset.csv')  # Reemplaza con la ruta a tu archivo CSV

# Seleccionar las características
X = data[['BMI', 'HighBP']]

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Configuraciones de DBSCAN
params = [
    (0.1, 5),   # Configuración 1
    (0.3, 5),   # Configuración 2
    (0.5, 5),   # Configuración 3
    (0.3, 10)   # Configuración 4
]

results = []

# Entrenar el modelo DBSCAN con diferentes configuraciones
for eps, min_samples in params:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, clusters) if len(set(clusters)) > 1 else -1
    results.append((eps, min_samples, clusters, silhouette))

# Visualizar los resultados
plt.figure(figsize=(12, 8))
for i, (eps, min_samples, clusters, _) in enumerate(results):
    plt.subplot(2, 2, i+1)
    plt.scatter(X['BMI'], X['HighBP'], c=clusters, cmap='viridis', marker='o', edgecolor='k')
    plt.title(f'DBSCAN: eps={eps}, min_samples={min_samples}')
    plt.xlabel('BMI')
    plt.ylabel('HighBP')

plt.tight_layout()
plt.show()

# Analizar los resultados
for eps, min_samples, clusters, silhouette in results:
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f'Configuración: eps={eps}, min_samples={min_samples}, '
          f'Número de clusters: {n_clusters}, Silhouette Score: {silhouette:.3f}')