pragma circom 2.0.3;
include "../../circuits/mlp.circom";

// n  = 1;
// d  = 784;
// h1 = 256;
// h2 = 128;
// o  = 10;
component main  {public [in, w1, b1, w2, b2, w3, b3]}  = MLP(1, 784, 256, 128, 10);
