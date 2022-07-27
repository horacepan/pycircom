import time
import torch
import torch.nn as nn
import numpy as np
from gen_fixed_net import *

def set_params(net, weights):
    for p, w in zip(net.parameters(), weights):
        p.data = torch.FloatTensor(w)

def main():
    st = time.time()
    np.random.seed(0)
    in_dim = 784
    out_dim = 10
    # tuples of (out_dim, in_dim)
    #fc_shapes = [(8, in_dim), (4, 8), (2, 4)]
    fc_shapes = [(256, in_dim), (128, 256)]
    out_shape = (10, 128)
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
        'name': 'mlp_big',
        'in_dim': in_dim,
        'out_dim': out_dim,
        'circ_args': "n",
        'layers': layers,
        #'expected': "64796788"
    }
    gen_net(spec)

    X = torch.ones(1, in_dim)
    net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    set_params(net, params)
    print("expected:", net(X))
    print("Elapsed: {:.2f}s".format(time.time() - st))

main()
