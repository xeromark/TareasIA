import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

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

### Declaracion de variables
estadoActual = [0, 0, 0, 0, 1, 0, 0, 0, 0]
temporal = [] #lista vacia para guardar los datos.
fs = [0,2,4,6,8] # Filas Seleccionadas para mostrar, en este caso G1, G3. G5, G7, G9

### Lógica del random-walk
for x in range(10):
   estadoActual = np.dot(estadoActual , Matriz) # Acumulacion del estado actual con los siguientes
   temporal.append(estadoActual) # Cada iteracion se alamcena su resultado acumulado

temporal = np.array(temporal) # Lo pasamos a una lista de array de tipo enteros no tuplas

# Se grafican los resultados obtenidos
for i in fs:
  plt.plot(temporal[: , i], label=f'Canción G{i+1}')

print(tabulate(temporal[:, fs], headers=["G1","G3","G5","G7","G9"], tablefmt="grid")) # Mostrar la tabla tabulada asociada al grafico 

plt.xlabel('Iteración')
plt.ylabel('Probabilidad (%)')
plt.title('Probabilidad de reproducción de música')
plt.legend()
plt.grid(True)
plt.show()