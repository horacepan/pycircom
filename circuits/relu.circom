pragma circom 2.0.4;

include "../node_modules/circomlib/circuits/sign.circom";
include "../node_modules/circomlib/circuits/bitify.circom";

template IsNegative() {
    signal input in;
    signal output out;

    component sign = Sign();
    component n2b = Num2Bits(254);
    n2b.in <== in;
    for (var i = 0; i < 254; i++) {
        sign.in[i] <== n2b.out[i];
    }
    out <== sign.sign;
}

template IsPositive() {
    signal input in;
    signal output out;

    component sign = Sign();
    component n2b = Num2Bits(254);
    n2b.in <== in;
    for (var i = 0; i < 254; i++) {
        sign.in[i] <== n2b.out[i];
    }
    out <== 1 - sign.sign;
}


template ReLU() {
    signal input in;
    signal output out;
    component pos = IsPositive();
    pos.in <== in;
    out <== in * pos.out;
}

template ReLU2d(m, n) {
    signal input in[m][n];
    signal output out[m][n];
    component pos[m][n];

    
    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            pos[m][n] = IsPositive();
            pos[m][n].in <== in[m][n];

            out[m][n] <== in[m][n] * pos[m][n].out;
        }
    }
}