import scipy.sparse
import numpy as np

SP_FMT_LINES = [
    "template {name} (n) {{",
    "    var DIN = {DIN};",
    "    var DOUT = {DOUT};",
    "    // precondition: rows are sorted",
    "    // vals[i] corresponds to the (rows[i], cols[i]) index of the sparse matrix W",
    "    var NNZ = {NNZ}; // number of non zeros",
    "    var rows[NNZ] = {rows};",
    "    var cols[NNZ] = {cols};",
    "    var vals[NNZ] = {vals};",
    "    var rnnz[DOUT] = {rnnz}; // index i of rnnz = number of non zeros in row i of sparse W",
    "    var bias[DOUT] = {bias};",
    "    signal input in[n][DIN];",
    "    signal output out[n][DOUT]; //store intermediate sums",
    "    signal _sums[n][DOUT][DIN+1];",
    "    var l = 0; // counter var",
    "    for (var i = 0 ; i < n; i++) {{",
    "        for (var j = 0 ; j < DOUT; j++) {{",
    "            _sums[i][j][0] <== 0;",
    "            // k = 0...rnnz in row [j]",
    "            // k is over the columns (over DIN)",
    "            for (var k = 0; k < rnnz[j]; k++) {{",
    "                _sums[i][j][k+1] <== _sums[i][j][k] + vals[l] * in[i][cols[l]];",
    "                l += 1;",
    "            }}",
    "            out[i][j] <== _sums[i][j][rnnz[j]] + bias[j];",
    "        }}",
    "    }}",
    "}}",
]
SP_FMT_STR = "\n".join(SP_FMT_LINES)

'''
Fill: name, DIN, DOUT, NNZ, rows, cols, vals, rnnz, bias
'''

def main():
    np.random.seed(2)
    '''
    matrix = [1 0 2]
             [0 1 0]
             [0 0 1]
    '''
    DIN = 123
    DOUT = 100
    x = scipy.sparse.random(DOUT, DIN, density=0.2, dtype=int)
    x.data = np.random.randint(0, 100, size=x.data.shape)
    NNZ = len(x.data)
    bias = np.random.randint(0, 100, size=(DOUT,))
    # need to compute this...
    rnnz = [0 for i in range(DOUT)]
    for r in x.row:
        rnnz[r] += 1
    assert(sum(rnnz) == NNZ)

    r = x.row.tolist()
    c = x.col.tolist()
    v = x.data.tolist()
    rows, cols, vals = zip(*sorted(zip(r, c, v)))
    rows, cols, vals = list(rows), list(cols), list(vals)
    fmt_dict = {
        'name': 'spmm',
        'DIN': DIN,
        'DOUT': DOUT,
        'NNZ': NNZ,
    	'rows': rows,
    	'cols': cols,
    	'vals': vals,
        'rnnz': rnnz,
    	'bias':  bias.tolist()
    }
    assert(len(fmt_dict['vals']) == NNZ)
    input = np.random.randint(0, 10, size=(DIN,1))
    X = x.toarray()
    result = X@input + bias.reshape(-1, 1)
    print(SP_FMT_STR.format(**fmt_dict))
    print('input:', input.ravel().tolist())
    print('result:', result.ravel().tolist())


if __name__ == '__main__':
    main()
