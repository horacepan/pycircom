import torch
import torch.nn as nn
import numpy as np
from gen_fixed_net import *

def set_params(net, weights):
    for p, w in zip(net.parameters(), weights):
        p.data = torch.FloatTensor(w)

def main():
    np.random.seed(0)
    in_dim = 16
    # tuples of (out_dim, in_dim)
    #fc_shapes = [(8, in_dim), (4, 8), (2, 4)]
    fc_shapes = [(8, in_dim), (4, 8)]
    out_shape = (1, 4)
    layers = []
    params = []
    for shape in fc_shapes:
        layers.append(
            {
                'type': 'fc',
                'weight': np.random.randint(0, 100, size=shape),
                'bias': np.random.randint(0, 100, size=(shape[0],))
            }
        )
        params.append(layers[-1]['weight'])
        params.append(layers[-1]['bias'])

        layers.append(
            {
                'type': 'relu2d'
            }
        )

    layers.append({
        'type': 'fc',
        'weight': np.random.randint(0, 100, size=out_shape),
        'bias': np.random.randint(0, 100, size=(out_shape[0],))
    })
    params.append(layers[-1]['weight'])
    params.append(layers[-1]['bias'])


    spec = {
        'version': "2.0.3",
        'name': 'mlp_small',
        'in_dim': in_dim,
        'out_dim': 1,
        'circ_args': "n",
        'layers': layers,
        'expected': "64796788"
    }
    gen_net(spec)

    X = torch.ones(1, in_dim)
    net = nn.Sequential(
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    set_params(net, params)
    print("expected:", net(X))

main()
