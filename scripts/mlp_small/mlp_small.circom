pragma circom 2.0.3;
include "./../../circuits/mlp_small.circom";

component main  {public [in]}  = mlp_small(1);
