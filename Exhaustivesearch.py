# Load library
import bnlearn as bn
from tabulate import tabulate
import pandas as pd

# Load example
df = pd.read_csv('dataset.csv')


# Structure learning
model = bn.structure_learning.fit(df, methodtype='ex', scoretype='bic')
# Compute edge strength using chi-square independence test and remove (prune) the not-signficant edges

model = bn.independence_test(model, df, alpha=0.05, prune=True)

# Examine the output of the chi-square test. All P values are significant. Nothing is removed.
print(tabulate(model['independence_test'], tablefmt="grid", headers="keys"))


# Grafico
bn.plot(model)