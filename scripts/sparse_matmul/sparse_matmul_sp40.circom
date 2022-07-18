pragma circom 2.0.4;

include "../../circuits/sparse_matmul_sp40.circom";

component main {public [A_vals]} = sparse_matmul(40, 40, 40, 100);