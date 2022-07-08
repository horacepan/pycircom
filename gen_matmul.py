import numpy as np

TAB_SPACE = 4
TAB = " " * TAB_SPACE

def tab_format_line(s, tabs):
    return ("\t" * tabs) + s

def gen_header(version="2.0.5"):
    '''
    '''
    return (f"pragma circom {version}") + "\n"

def gen_matvec(shape_A, shape_x, A_name):
    '''
    Generate circom code for matrix vector multiplication

    template matmul_vec(m, n) {
        signal input A[m][n];
        signal input x[n];
        signal output[m];

        for (var i=0; i<m; i++) {
            for (var j=0; j<n; j++) {

            }
        }
    }
    '''
    r, c = shape_A
    return s

# Write the full thing out
def gen_matmul(A, B):
    '''
    A, B are dummy matrices (entries dont matter)
    '''
    m, p = A.shape
    p, n = B.shape

    res = ""
    res += gen_header();

    res += "template matmul(m, p, n) {\n"
    res += TAB + "signal input A[m][p];\n"
    res += TAB + "signal input B[p][n];\n"
    res += TAB + "signal output out[m][n];\n"

    '''
    For each row i with non zero A entries construct a signal
    signal t{i}[n][l_i + 1];
    where l_i is the number of non-zero entries of A.
    Post condition:
    c{i}[n][l_i] stores the value of Out[i][j][l_i] stores the value of [AB]_{ij}
    for j = 1, ..., n
    [AB]_{ij} = \sum_{k} A_{ik} B_{kj}
    '''
    # since this is a FULL matmul, we have all the entries
    # declare the signals
    res += "\n"
    for i in range(m):
        res += TAB + f"signal c{i}[n][p];" + "\n"
    res += "\n"

    # outer 2 loop for iteratively filling c[i][j]
    for i in range(m):
        for j in range(n):
            cij = []
            cij.append(f"c{i}[{j}][0] <== A[{i}][0] * B[0][{j}];")
            for k in range(1, p):
                cij.append(f"c{i}[{j}][{k}] <== c{i}[{j}][{k-1}] + A[{i}][{k}] * B[{k}][{j}];")

            res += TAB + (" ".join(cij)) + "\n"

    # fill the output
    res += "\n"
    for i in range(m):
        x = []
        for j in range(n):
            x.append(f"out[{i}][{j}] <== c{i}[{j}][{p-1}];")
        res += TAB + " ".join(x) + "\n"

    res += "}"
    return res

def gen_sparse_matmul():
    '''
    Return circom code for A@B where
    A: sparse m x p matrix
    B: sparse p x n matrix
    '''
    pass

if __name__ == '__main__':
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    print(gen_matmul(A, B))
