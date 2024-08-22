import pandas as pd
import bnlearn as bn

# Cargar el archivo CSV
df = pd.read_csv('./dataset.csv')


model = bn.structure_learning.fit(df)

plot = bn.plot(model)