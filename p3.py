import numpy as np

# Definir matrices de transición y emisión
Matriz_T = np.array([
    [0.2, 0.6, 0.2],
    [0.3, 0.0, 0.7],
    [0.5, 0.0, 0.5]
])

Matriz_E = np.array([
    [0.8, 0.2],  # Estado 1: P(Satisfecho | q1) = 0.8, P(Insatisfecho | q1) = 0.2
    [0.3, 0.7],  # Estado 2: P(Satisfecho | q2) = 0.3, P(Insatisfecho | q2) = 0.7
    [0.6, 0.4]   # Estado 3: P(Satisfecho | q3) = 0.6, P(Insatisfecho | q3) = 0.4
])

# Secuencia de observaciones (0 = Satisfecho, 1 = Insatisfecho)
observaciones = [0, 1, 1, 0, 1, 1, 1, 1, 0]

# Probabilidades iniciales uniformes π 
#               R    S    C
π = np.array([1/3, 1/3, 1/3])

# Número de estados y observaciones
n_estados = Matriz_T.shape[0]
n_obs = len(observaciones)

# Matriz para almacenar las probabilidades (Viterbi)
#                       3   x   9
viterbi = np.zeros((n_estados, n_obs))


# Matriz para almacenar los pasos (para backtracking)
backpointer = np.zeros((n_estados, n_obs), dtype=int)

# Inicialización
#                   3
for s in range(n_estados):
#   μ0(q0) =        P(q0) * P(o1/q0)
    viterbi[s, 0] = π[s] * Matriz_E[s, observaciones[0]]
    backpointer[s, 0] = 0

# Ejemplo para la inicialización
# s = 0 => viterbi[0, 0] = pi[0] * Matriz_E[0, observaciones[0]] =>  viterbi[0, 0] = 1/3 * 0.8 = 0,26
# s = 1 => viterbi[1, 0] = pi[1] * Matriz_E[1, observaciones[0]] =>  viterbi[1, 0] = 1/3 * 0.3 = 0,1
# s = 2 => viterbi[2, 0] = pi[2] * Matriz_E[2, observaciones[0]] =>  viterbi[2, 0] = 1/3 * 0.6 = 0,2

# Recursión         3
for t in range(1, n_obs):
#                      9
    for s in range(n_estados):
#                           μk−1(qk−1)    * P(qk /qk−1)
        prob_transition = viterbi[:, t-1] * Matriz_T[:, s]
#        print(viterbi[:, t-1] ,"*", Matriz_T[:, s])
#        print(prob_transition)
#          μk(qk) =                      P(ok/qk) * Max[ μk−1(qk−1) * P(qk /qk−1) ]
        viterbi[s, t] = Matriz_E[s, observaciones[t]] * np.max(prob_transition)
        backpointer[s, t] = np.argmax(prob_transition)


# Backtracking para encontrar la secuencia de estados más probable
best_path = np.zeros(n_obs, dtype=int)
best_path[-1] = np.argmax(viterbi[:, -1])  # Último estado más probable

for t in range(n_obs-2, -1, -1):
    best_path[t] = backpointer[best_path[t+1], t+1]

# Probabilidad del estado más probable en el tiempo t=5 (q5)
q5_prob = viterbi[:, 4]
best_q5_state = np.argmax(q5_prob) + 1  # El estado más probable en t=5 (q5)

"""
# Imprimir resultados
print("La probabilidad en t=5 para cada estado:", q5_prob)
print("El estado más probable en t=5 es q:", best_q5_state)
print("La secuencia más probable de estados:", best_path + 1)
"""
