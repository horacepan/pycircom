template square() {
    signal input in;
    signal output out;
    out <== in * in;
}

template square2d(m, n) {
    signal input in[m][n];
    signal output out[m][n];

    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            out[i][j] <== in[i][j] * in[i][j];
        }
    }
}
