import json
import numpy as np
import torch
import torch.nn as nn

from gen_inputs import serialize_list, serialize_numpy
def nparams(net):
    tot = 0
    for p in net.parameters():
        tot += p.numel()
    return tot

def serialize_linear(x):
    w = serialize_numpy(x.weight.data.round().int().numpy().T)
    b = serialize_numpy(x.bias.data.round().int().numpy())
    return {'weight': w, 'bias': b}

def gen_mlp(layers):
    d = {}
    cnt = 0
    for idx, l in enumerate(layers):
        if type(l) == nn.Linear:
            cnt += 1
            ld = serialize_linear(l)
            d[f'w{cnt}'] = ld['weight']
            d[f'b{cnt}'] = ld['bias']

    return d

def gen_example():
    torch.manual_seed(0)
    layers = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    #layers = nn.Sequential(
    #    nn.Linear(5, 2),
    #    nn.ReLU(),
    #    nn.Linear(2, 4)
    #)

    for l in layers:
        if type(l) == nn.Linear:
            w = l.weight
            b = l.bias
            w.data *= 100
            b.data *= 100
            w.data.round_()
            b.data.round_()

    din = {'in': serialize_numpy(np.ones((1,784)))}
    mlp = gen_mlp(layers)
    din.update(mlp)

    output = layers(torch.ones(1,784))
    din['exp_out'] = serialize_numpy(output.data.numpy())
    print(nparams(layers))
    with open('../scripts/mlp3/input_mlp3.json', 'w') as f:
        json.dump(din, f, indent=2)

if __name__ == '__main__':
    gen_example()
