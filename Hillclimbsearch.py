# Load library
import bnlearn as bn
import pandas as pd
from tabulate import tabulate

df = pd.read_csv('dataset.csv')

modelhc = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')

modelhc = bn.independence_test(modelhc, df, alpha=0.05, prune=False)
bn.plot(modelhc)

print(tabulate(modelhc['independence_test'], headers="keys"))




parametrohc = bn.parameter_learning.fit(modelhc, df)
inferenciahc = bn.inference.fit(parametrohc, variables=['Hierba_mojada'], evidence={'Lluvia': 1})

# Mostrar el resultado de la inferencia
print(inferenciahc)


parametrohc2 = bn.parameter_learning.fit(modelhc, df)
inferenciahc2 = bn.inference.fit(parametrohc2, variables=['Aspersor'], evidence={'Nublado': 1})

# Mostrar el resultado de la inferencia
print(inferenciahc2)