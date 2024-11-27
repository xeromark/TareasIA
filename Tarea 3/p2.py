import numpy as np
from feed_forward_network.feedforward import FeedForward
from neuron.neuron_nn import NeuronNN
from neuron.neuron_kan import NeuronKAN
import matplotlib.pyplot as plt
from utils.activations import relu
from utils.edge_fun import get_bsplines

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
data = pd.read_csv('./adult.data', header=None, names=columns, skipinitialspace=True)

x = data[['age']].values  # Usar 'age' como característica
y = data[['hours-per-week']].values  # Usar 'hours-per-week' como objetivo

x_train, _, y_train, _ = train_test_split(x, y, train_size=442, random_state=42)

# Estandarizar los datos
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
y_train = scaler_y.fit_transform(y_train)

color_plots = {'dataset': 'b', 'kan': 'orange', 'mlp': 'green'}

n_iter_train_1d = 500
loss_tol_1d = 0.05
seed = 476

# KAN training
kan_1d = FeedForward(
    [1, 2, 2, 1],  # layer size
    eps=0.01,  # gradient descent parameter
    n_weights_per_edge=7,  # n. edge functions
    neuron_class=NeuronKAN,
    x_bounds=[-1, 1],  # input domain bounds
    get_edge_fun=get_bsplines,  # edge function type (B-splines)
    seed=seed,
    weights_range=[-1, 1]
)
kan_1d.train(x_train, y_train, n_iter_max=n_iter_train_1d, loss_tol=loss_tol_1d)

# MLP training
mlp_1d = FeedForward(
    [1, 13, 1],  # layer size
    eps=0.005,  # gradient descent parameter
    activation=relu,  # activation type (ReLU)
    neuron_class=NeuronNN,
    seed=seed,
    weights_range=[-0.5, 0.5]
)
mlp_1d.train(x_train, y_train, n_iter_max=n_iter_train_1d, loss_tol=loss_tol_1d)

# Regression on training data
fig, ax = plt.subplots(figsize=(4, 3.2))
x_plot = np.linspace(x_train.min(), x_train.max(), 1000).reshape(-1, 1)
ax.plot(x_train, y_train, 'o', color=color_plots['dataset'], label='training dataset')
ax.plot(x_plot, [kan_1d(x) for x in x_plot], color=color_plots['kan'], label='KAN')
ax.plot(x_train, [kan_1d(x) for x in x_train], 'x', color=color_plots['kan'], fillstyle='none')
ax.plot(x_plot, [mlp_1d(x) for x in x_plot], color=color_plots['mlp'], label='MLP')
ax.plot(x_train, [mlp_1d(x) for x in x_train], 'd', color=color_plots['mlp'], fillstyle='none')
ax.set_xlabel('Age (normalized)', fontsize=13)
ax.set_title('Regression on Adults Dataset', fontsize=15)
ax.legend()
ax.grid()
fig.tight_layout()
plt.show()
