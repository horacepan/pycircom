pragma circom 2.0.4;
include "relu.circom";
include "fc.circom";

template MLP(n, d, h1, o) {
    signal input in[n][d];
    // fc 1
    signal input w1[d][h1];
    signal input b1[h1];
    // fc 2
    signal input w2[h1][o];
    signal input b2[o];
    signal input exp_out[n][o];
    signal output out[n][o];

    component fc1 = fc(n, d, h1);
    component fc2 = fc(n, h1, o);
    component relu1[n][h1];
    for (var i=0; i<n; i++) {
        for (var j=0; j<h1; j++) {
            relu1[i][j] = ReLU();
        }
    }

    // fill the inputs of fc1
    for (var i=0; i<n; i++) {
        for (var j=0; j<d; j++) {
            fc1.in[i][j] <== in[i][j];
        }
    }

    // fill the params of fc1
    for (var j=0; j<h1; j++) {
        for (var i=0; i<d; i++) {
            fc1.weight[i][j] <== w1[i][j];
        }
        fc1.bias[j] <== b1[j];
    }

    // Do the first relu
    for (var i=0; i<n; i++) {
        for (var j=0; j<h1; j++) {
            relu1[i][j].in <== fc1.out[i][j];
        }
    }
    // fill the inputs of fc2
    for (var i=0; i<n; i++) {
        for (var j=0; j<h1; j++) {
            fc2.in[i][j] <== relu1[i][j].out;
        }
    }

    // fill the params of fc2
    for (var j=0; j<o; j++) {
        for (var i=0; i<h1; i++) {
            fc2.weight[i][j] <== w2[i][j];
        }
        fc2.bias[j] <== b2[j];
    }

    // Fill the output: n x o
    for (var i=0; i<n; i++) {
        for (var j=0; j<o; j++) {
            out[i][j] <== fc2.out[i][j];
            out[i][j] === exp_out[i][j];
        }
    }
}
