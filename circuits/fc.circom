pragma circom 2.0.4;

// Evaluate X @ W + b
// X: n x d1
// W: d1 x d2
// b: d2
template fc (n, d1, d2) {
    signal input in[n][d1];
    signal input weight[d1][d2];
    signal input bias[d2];
    signal output out[n][d2];

    signal sums[n][d2][d1];
    for (var i=0; i<n; i++) {
        for (var j=0; j<d2; j++) {
            sums[i][j][0] <== in[i][0] * weight[0][j];
            for (var k=1; k<d1; k++) {
                sums[i][j][k] <== sums[i][j][k-1] + in[i][k] * weight[k][j];
            }
            out[i][j] <== sums[i][j][d1-1] + bias[j];
        }
    }
}