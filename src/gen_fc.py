import numpy as np

FC_FMT_STR = "template {name} (n) {{\n\
    var DIN = {DIN};\n\
    var DOUT = {DOUT};\n\
    var weight[DOUT][DIN] = {weight};\n\
    var bias[DOUT] = {bias};\n\n\
    signal input in[n][DIN];\n\
    signal output out[n][DOUT];\n\n\
    signal _sums[n][DOUT][DIN+1];\n\
    for (var i = 0; i < n; i++) {{\n\
        for (var k = 0; k < DOUT; k++) {{\n\
            _sums[i][k][0] <== 0;\n\
            for (var j = 0; j < DIN; j++) {{\n\
                _sums[i][k][j+1] <== _sums[i][k][j] + weight[k][j] * in[i][j];\n\
            }}\n\
            out[i][k] <== _sums[i][k][DIN] + bias[k];\n\
        }}\n\
    }}\n\
}}"

FC_INPUT_FMT ="\
component fc_{_id} = fc({n});\n\
for (var i = 0;  i < {m}; i++) {{\n\
    for (var j = 0; j < {n}; j++) {{\n\
        fc_{_id}.in <== {signal_in}.in\n\
    }}\n\
}}\n"
def gen_fc(weight, bias, name='_fc'):
    '''
    Produces a circuit with the values of weight and bias hardcoded in
    as variables.
    '''
    DOUT = len(weight)
    DIN = len(weight[0])
    weight_str = (weight)
    bias_str = (bias)
    fmt_vals = {
        "name": name,
        "DIN": DIN,
        "DOUT": DOUT,
        "weight": weight.tolist(),
        "bias": bias.tolist()
    }
    circuit = FC_FMT_STR.format(**fmt_vals)
    return circuit

def fc_input_str(signal_in, dims):
    m, n = dims

if __name__ == '__main__':
    weight = [[1,0], [0,1], [1,1]]
    bias = [10,20,30]
    print(gen_fc(weight, bias))
