import scipy.sparse
import numpy as np

SP_FMT_LINES = [
    "template {name} (batch_size) {{",
    "    var DIN = {DIN};",
    "    var DOUT = {DOUT};",
    "    // precondition: rows are sorted",
    "    // vals[i] corresponds to the (rows[i], cols[i]) index of the sparse matrix W",
    "    var NNZ = {NNZ}; // number of non zeros",
    "    var rows[NNZ] = {rows};",
    "    var cols[NNZ] = {cols};",
    "    var vals[NNZ] = {vals};",
    "    var rnnz[DOUT] = {rnnz}; // rnnz[i] = number of non zeros in row i of the sparse matrix W",
    "    var bias[DOUT] = {bias};",
    "    signal input in[batch_size][DIN];",
    "    signal output out[batch_size][DOUT]; //store intermediate sums",
    "    signal _sums[batch_size][DOUT][DIN+1];",
    "    var l = 0; // counter var",
    "    for (var i = 0 ; i < batch_size; i++) {{",
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
    "        l = 0;",
    "    }}",
    "}}",
]
SP_FMT_STR = "\n".join(SP_FMT_LINES)

'''
Fill: name, DIN, DOUT, NNZ, rows, cols, vals, rnnz, bias
'''

def gen_sparse_fc(sparse_mat, bias, template_name='spmm'):
    DOUT, DIN =  sparse_mat.shape
    assert bias.shape == (DOUT,)
    NNZ = len(sparse_mat.data)

    # rnnz[i] = num nnz in row i of sparse_mat
    rnnz = [0 for i in range(DOUT)]
    for r in sparse_mat.row:
        rnnz[r] += 1
    assert(sum(rnnz) == NNZ)

    r = sparse_mat.row.tolist()
    c = sparse_mat.col.tolist()
    v = sparse_mat.data.tolist()
    rows, cols, vals = zip(*sorted(zip(r, c, v)))
    rows, cols, vals = list(rows), list(cols), list(vals)
    fmt_dict = {
        'name': template_name,
        'DIN': DIN,
        'DOUT': DOUT,
        'NNZ': NNZ,
    	'rows': rows,
    	'cols': cols,
    	'vals': vals,
        'rnnz': rnnz,
    	'bias':  bias.tolist()
    }

    assert(len(rows) == NNZ)
    assert(len(cols) == NNZ)
    assert(len(vals) == NNZ)
    return SP_FMT_STR.format(**fmt_dict)

def test_gen_sparse():
    np.random.seed(0)
    batch_size = 2
    DIN = 5
    DOUT = 4
    template_name = 'spmm'
    sparse_mat = scipy.sparse.random(DOUT, DIN, density=0.2, dtype=int)
    sparse_mat.data = np.random.randint(0, 100, size=sparse_mat.data.shape)
    bias = np.random.randint(0, 100, size=(DOUT,))
    input = np.random.randint(0, 10, size=(DIN, batch_size))
    output = sparse_mat.todense()@input + bias.reshape(-1, 1)

    circuit_str = gen_sparse_fc(sparse_mat, bias, template_name)
    print(circuit_str)
    print('component main { public [ in ] } = ' + f'{template_name}({batch_size});')
    print('/* INPUT = {')
    print('    "in": {}'.format(np.array2string(input.T, separator=',')))
    print('} */')
    print('/*Expected output:\n{}\n*/'.format(np.array2string(output, separator=',')))

if __name__ == '__main__':
    test_gen_sparse()
