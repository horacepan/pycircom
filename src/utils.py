import numpy as np
import json

def stringify_list(lst):
    return "[{}]".format(",".join([stringify_num(x) for x in lst]))

def stringify_num(x):
    return "\"{}\"".format(x)

def stringify_matrix(mat):
    return [stringify_list(row) for row in mat]

def gen_input(js, f):
    x = {}
    print("{", file=f)
    end = ","
    for i, (k, v) in enumerate(js.items()):
        if i == len(js) - 1:
            # dont add comma
            end = ""
        if type(v) == np.ndarray:
            x[k] = v.astype(int).tolist()
            print("  \"{}\": {}{}".format(k, v.astype(int).tolist(), end), file=f)
        else:
            x[k] = v
            print("  \"{}\": {}{}".format(k, v, end), file=f)

    print("}", file=f)

if __name__ == "__main__":
    np.random.seed(1)
    js = {
        "A_vals": np.random.randint(0, 100, (4,)),
        "B": np.random.randint(0, 100, (3,3))
    }
    gen_input(js)
