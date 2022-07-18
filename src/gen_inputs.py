import json
import numpy as np

P = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def serialize_mlp(net):
    d = {}
    for l in net.layers:
        pass

def sanitize_neg(x):
    if x < 0:
        return str(P + x)
    else:
        return str(x)

def serialize_numpy(x):
    xob = x.astype(int).astype(object)
    sanitize = np.vectorize(sanitize_neg)
    return sanitize(xob).tolist()

def serialize_list(lst):
    return [sanitize_neg(x) for x in lst]

def serialize_scalar(x):
    return sanitize_neg(x)

def gen_input(d):
    out = {}

    for k, v in d.items():
        if type(v) == np.ndarray:
            out[k] = serialize_numpy(v)
        elif type(v) == list:
            out[k] = serialize_list(v)
        else:
            out[k] = serialize_scalar(v)

    return out

if __name__ == '__main__':
    d = {
        'x': -1,
        'w1': np.random.randint(-1, 3, size=(3,3)),
        'b1': np.ones(3)
    }
    print(json.dumps(gen_input(d)))
