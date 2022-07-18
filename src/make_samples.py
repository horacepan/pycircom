import numpy as np
import scipy.sparse as sparse

from utils import gen_input
from gen_matmul import gen_sparse_matmul

np.set_string_function(lambda x: repr(x).replace('(', '')
                                        .replace(')', '')
                                        .replace('array', '')
                                        .replace("       ", ' ') , repr=False)

def main(n):
    n = 5
    k = 10 
    name = f'sp{n}'
    fname = f'../scripts/sparse_matmul/{name}.npz'

    try:
        A = sparse.load_npz(fname)
    except:
        data = np.random.randint(0, 100, size=(k,))
        cs = np.random.choice(n * n, size=(k), replace=False)
        rows = [i // n for i in cs]
        cols = [i % n for i in cs]
        A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        sparse.save_npz(fname, A)

    B = np.ones(A.shape)
    js = {"A_vals": A.data, "B": B}

    with open(f'../build/sparse_matmul_sp{n}/AB_n{n}_k{k}.txt', 'w') as f:
        lst = (A@B).tolist()
        for row in lst:
            f.write(str(row))
            f.write("\n")

    with open(f'../circuits/sparse_matmul_{name}.circom', 'w') as f:
        f.write(gen_sparse_matmul(A, B))

    gen_input(js, open(f'../scripts/sparse_matmul/input_sparse_matmul_{name}.json', 'w'))

if __name__ == '__main__':
    main(40)