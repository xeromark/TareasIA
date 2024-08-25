# Load library
import bnlearn as bn
import pandas as pd
from tabulate import tabulate

# Load example
df = pd.read_csv('dataset.csv')

# Structure learning
model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
# Plot detected DAG
G = bn.plot(model)

# Compute edge strength using chi-square independence test
model1 = bn.independence_test(model, df, alpha=0.05, prune=False)
bn.plot(model1, pos=G['pos'])

# Examine the output of the chi-square test. 53 edges are detected but not all P values are significant, i.e. those with stat_test=False
print(tabulate(model1['independence_test'], headers="keys"))

# Compute edge strength using chi-square independence test and remove (prune) the not-signficant edges
model3 = bn.independence_test(model, df, alpha=0.05, prune=True)
bn.plot(model3, pos=G['pos'])