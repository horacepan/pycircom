pragma circom 2.0.3;
include "./../../circuits/mlp2.circom";

// n  = 1;
// d  = 3;
// h1 = 2;
// o  = 4;
component main  {public [in, w1, b1, w2, b2]}  = MLP(1, 5, 2, 4);