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
#    source     target     stat_test        p_value    chi_square    dof
#--  ---------  ---------  -----------  -----------  ------------  -----
# 0  Cloudy     Rain       True         1.08061e-87       394.062      1
# 1  Cloudy     Sprinkler  True         8.38371e-53       233.906      1
# 2  Rain       Wet_Grass  True         3.88651e-64       285.902      1
# 3  Sprinkler  Wet_Grass  True         1.19692e-23       100.478      1

# Plot
bn.plot(model)