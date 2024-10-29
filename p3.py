import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar tu dataset
# df = pd.read_csv('tu_dataset.csv')  # Asegúrate de cargar tu dataset correctamente

# Seleccionar características
X = df[['HighBP', 'BMI']]

# Escalar las características
X_scaled = StandardScaler().fit_transform(X)

# Configuraciones de DBSCAN
configs = [
    {'eps': 0.3, 'min_samples': 5},
    {'eps': 0.5, 'min_samples': 5},
    {'eps': 0.3, 'min_samples': 10},
    {'eps': 0.5, 'min_samples': 10}
]

# Entrenar y graficar resultados para cada configuración
for config in configs:
    dbscan = DBSCAN(eps=config['eps'], min_samples=config['min_samples'])
    clusters = dbscan.fit_predict(X_scaled)

    # Graficar resultados
    plt.figure()
    plt.title(f'DBSCAN: eps={config["eps"]}, min_samples={config["min_samples"]}')
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', marker='o')
    plt.xlabel('HighBP')
    plt.ylabel('BMI')
    plt.colorbar(label='Cluster Label')
    plt.grid()
    plt.show()