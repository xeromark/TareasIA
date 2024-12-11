import numpy as np
from tabulate import tabulate

# Este metodo por python para calcular la distribucion estacionaria de la matriz, 
# se basa en el siguiente proceso: 
#  πP = π
#  πP - π = 0
#  π(P - I) = 0  // Donde I es la matriz identidad, se distribuyó a partir de las propiedades de las matrices
#
# Luego a partir de esto se encontraron los resultados

Matriz = np.array([
   #  G1    G2    G3    G4    G5    G6    G7    G8    G9
    [0.25, 0.06, 0.08, 0.15, 0.04, 0.02, 0.15, 0.15, 0.10],  # G1
    [0.15, 0.15, 0.10, 0.22, 0.01, 0.02, 0.15, 0.10, 0.10],  # G2
    [0.12, 0.00, 0.05, 0.24, 0.14, 0.04, 0.27, 0.07, 0.07],  # G3
    [0.05, 0.13, 0.05, 0.30, 0.10, 0.10, 0.22, 0.05, 0.00],  # G4
    [0.18, 0.20, 0.07, 0.20, 0.15, 0.05, 0.05, 0.05, 0.05],  # G5
    [0.20, 0.10, 0.20, 0.05, 0.05, 0.10, 0.02, 0.15, 0.13],  # G6
    [0.01, 0.05, 0.15, 0.14, 0.17, 0.10, 0.12, 0.10, 0.16],  # G7
    [0.17, 0.15, 0.07, 0.07, 0.15, 0.10, 0.12, 0.09, 0.08],  # G8
    [0.13, 0.11, 0.13, 0.03, 0.20, 0.20, 0.04, 0.15, 0.01]   # G9
])


# Restar la matriz identidad de P
A = Matriz.T - np.eye(Matriz.shape[0])

# Añadir la restricción de que la suma de los elementos de π sea 1
A = np.vstack([A, np.ones(Matriz.shape[0])])

# Vector de ceros, excepto para la restricción de la suma
b = np.zeros(Matriz.shape[0])
b = np.append(b, 1)

# Resolver el sistema de ecuaciones
Gn = np.linalg.lstsq(A, b, rcond=None)[0]

print("La distribución estacionaria es:" )

for i in range(9):
    print(f"G{i+1}" , "=", Gn[i])


