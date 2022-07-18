pragma circom 2.0.4;

template matmul (m, p, n) {
    signal input A[m][p];
    signal input B[p][n];
    signal output out[m][n];
    
    signal sums[m][p][n];
    // C_{ij} = \sum_{k=1}^p A_{ik} B_{kj}
    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            sums[i][0][j] <== A[i][0] * B[0][j];
            for (var k = 1; k < p; k++) {
                sums[i][k][j] <== sums[i][k-1][j] + A[i][k] * B[k][j];
            }

            out[i][j] <== sums[i][p-1][j];
        }
    }
}