# Load library
import bnlearn as bn
import pandas as pd
from tabulate import tabulate

# Load example
df = pd.read_csv('dataset.csv')

# Structure learning
modelhc = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
# Plot detected DAG

# Compute edge strength using chi-square independence test
modelhc = bn.independence_test(modelhc, df, alpha=0.05, prune=False)
bn.plot(modelhc)

# Examine the output of the chi-square test. 53 edges are detected but not all P values are significant, i.e. those with stat_test=False
print(tabulate(modelhc['independence_test'], headers="keys"))




parametrohc = bn.parameter_learning.fit(modelhc, df)
inferenciahc = bn.inference.fit(parametrohc, variables=['Hierba_mojada'], evidence={'Lluvia': 1})

# Mostrar el resultado de la inferencia
print(inferenciahc)


parametrohc2 = bn.parameter_learning.fit(modelhc, df)
inferenciahc2 = bn.inference.fit(parametrohc2, variables=['Aspersor'], evidence={'Nublado': 1})

# Mostrar el resultado de la inferencia
print(inferenciahc2)