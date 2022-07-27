pragma circom 2.0.3;
include "./../../circuits/mlp_big.circom";

component main  {public [in]}  = mlp_big(1);
