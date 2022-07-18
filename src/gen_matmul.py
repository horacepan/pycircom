import numpy as np
import scipy.sparse as sparse
from utils import gen_input

VERSION = "2.0.4"
TAB_SPACE = 4
TAB = " " * TAB_SPACE

def gen_header(version):
    return (f"pragma circom {version};" + "\n")

# Write the full thing out
def gen_matmul(A, B):
    '''
    A, B are dummy matrices (entries dont matter)
    '''
    m, p = A.shape
    p, n = B.shape

    res = ""
    res += gen_header(VERSION)

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

def gen_sparse_matmul(A, B):
    '''
    Fixed model A?
    A: m x p, B: p x n -> C = m x n

    Generate circom code that computes A@B where
    A is a k-sparse matrix of shape (m, p), and B is a dense matrix of size (p, n)
    - k: the number of non-zero values in A
    - A_vals: the k non-zero values of A
    - B: dense matrix

    Precondition: entries of A_vals must be sorted by row, then column
    '''
    m, p = A.shape
    p, n = B.shape

    res = ""
    res += gen_header(VERSION)

    res += "template sparse_matmul(m, p, n, k) {\n"
    res += TAB + "signal input A_vals[k];\n"
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
    rows, cols = A.nonzero()
    vals = A.data
    rcs = sorted(zip(rows, cols))

    # group the entries into uniques
    row_vals = {}
    for r, c in rcs:
        if r not in row_vals:
            row_vals[r] = []
        row_vals[r].append(c)

    # declare the signals for accumulating the dot products of each row
    # dense B so we have n columns to evaluate for each row
    for r, cs in row_vals.items():
        res += TAB + f"signal c{r}[n][{len(cs)}];" + "\n"

    # evaluate dot product for entry [out]_{ij}
    idx = 0 # index into A_vals
    for i, cs in row_vals.items():
        # evaluate the contribution of each nonzero entry
        for j in range(n):
            cij = []
            cij.append(f"c{i}[{j}][0] <== A_vals[{idx}] * B[{cs[0]}][{j}];") # first entry of dot product

            for _ik in range(1, len(cs)):
                k = cs[_ik]
                cij.append(f"c{i}[{j}][{_ik}] <== c{i}[{j}][{_ik-1}] + A_vals[{idx + _ik}] * B[{k}][{j}];")
            res += TAB + " ".join(cij) + "\n"

        idx += len(cs)

    # fill the output. Output is still a m x n matrix but will be k-row sparse.
    res += "\n"
    for i in range(m):
        x = []
        if i not in rows:
            x.append(f"out[{i}][{j}] <== 0;")
            continue

        for j in range(n):
            k = len(row_vals[i]) - 1
            x.append(f"out[{i}][{j}] <== c{i}[{j}][{k}];")
        res += TAB + " ".join(x) + "\n"

    res += "}"
    return res
