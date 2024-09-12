import numpy as np
from tabulate import tabulate

def viterbi(O, A, B, π):
   
    # O: Secuencia de observaciones (lista de índices de observaciones).
    # A: Matriz de probabilidad de transición.
    # B: Matriz de probabilidad de emisión.
    # π: Vector de probabilidad iniciales que para este caso todas son iguales.

    n_estados = A.shape[0]  # Numero de estados
    n_obs = len(O)          # Numero de observaciones

    # 1. Inicialización
    #               9   x    3
    V = np.zeros((n_obs, n_estados))

    for s in range(n_estados):
        V[0, s] = π[s] * B[s, O[0]]

    # 2. Recursión
    for t in range(1, n_obs):
        for s in range(n_estados):
            prob = V[t-1] * A[:, s] * B[s, O[t]]
            V[t, s] = np.max(prob)
    
    return V

# Definir matrices de transición y emisión
# R: Ramen
# S: Salmorejo
# C: Cebolla

Matriz_T = np.array([
    # R    S    C
    [0.2, 0.6, 0.2], # R
    [0.3, 0.0, 0.7], # S
    [0.5, 0.0, 0.5]  # C
])

Matriz_E = np.array([
    # S    I
    [0.8, 0.2],
    [0.3, 0.7],
    [0.6, 0.4] 
])

# Probabilidades iniciales
π = np.array([1/3, 1/3, 1/3])  # Suponiendo una distribución inicial uniforme

# Secuencia de observaciones (1 = satisfecho, 0 = insatisfecho)
#observaciones = [1, 0, 0, 1, 0, 0, 0, 0, 1]
observaciones = [0, 1, 1, 0, 1, 1, 1, 1, 0]

# Ejecutar el algoritmo de Viterbi
V = viterbi(observaciones, Matriz_T, Matriz_E, π)

# Obtener la probabilidad en la quinta iteración (índice 4)
q5 = np.max(V[4])

    #best_path_prob = np.max(V[-1])
    #best_last_state = np.argmax(V[-1])
print(tabulate(V))
print("Probabilidad del estado oculto q5 más probable es", np.argmax(V[4]) , "con una probabilidad de", q5)
