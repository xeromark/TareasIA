# Instalar librerías necesarias

    pip install bnlearn

# Se importan las librerías necesarias:

    import pandas as pd
    import bnlearn as bn

# Se usa el siguiente código:

Exhaustivesearch determina todas las posibles estructuras de red bayesiana en un conjunto de datos para seleccionar la estructura que se adecue mejor estos. Por lo tanto, este método garantiza encontrar la estructura más óptima a costa de usar muchos recursos computacionales de la máquina en donde se utiliza.

```

import bnlearn as bn
from tabulate import tabulate
import pandas as pd

df = pd.read_csv('dataset.csv')

model = bn.structure_learning.fit(df, methodtype='ex', scoretype='bic')

model = bn.independence_test(model, df, alpha=0.05, prune=True)

print(tabulate(model['independence_test'], tablefmt="grid", headers="keys"))
bn.plot(model)

 ```

