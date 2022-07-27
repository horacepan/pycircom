pragma circom 2.0.3;

include "./relu.circom";
include "./fc.circom";


template mlp_small_fc1 (n) {
    var DIN = 16;
    var DOUT = 8;
    var weight[DOUT][DIN] = [[44, 47, 64, 67, 67, 9, 83, 21, 36, 87, 70, 88, 88, 12, 58, 65], [39, 87, 46, 88, 81, 37, 25, 77, 72, 9, 20, 80, 69, 79, 47, 64], [82, 99, 88, 49, 29, 19, 19, 14, 39, 32, 65, 9, 57, 32, 31, 74], [23, 35, 75, 55, 28, 34, 0, 0, 36, 53, 5, 38, 17, 79, 4, 42], [58, 31, 1, 65, 41, 57, 35, 11, 46, 82, 91, 0, 14, 99, 53, 12], [42, 84, 75, 68, 6, 68, 47, 3, 76, 52, 78, 15, 20, 99, 58, 23], [79, 13, 85, 48, 49, 69, 41, 35, 64, 95, 69, 94, 0, 50, 36, 34], [48, 93, 3, 98, 42, 77, 21, 73, 0, 10, 43, 58, 23, 59, 2, 98]];
    var bias[DOUT] = [62, 35, 94, 67, 82, 46, 99, 20];

    signal input in[n][DIN];
    signal output out[n][DOUT];

    signal _sums[n][DOUT][DIN+1];
    for (var i = 0; i < n; i++) {
        for (var k = 0; k < DOUT; k++) {
            _sums[i][k][0] <== 0;
            for (var j = 0; j < DIN; j++) {
                _sums[i][k][j+1] <== _sums[i][k][j] + weight[k][j] * in[i][j];
            }
            out[i][k] <== _sums[i][k][DIN] + bias[k];
        }
    }
}

template mlp_small_fc2 (n) {
    var DIN = 8;
    var DOUT = 4;
    var weight[DOUT][DIN] = [[81, 50, 27, 14, 41, 58, 65, 36], [10, 86, 43, 11, 2, 51, 80, 32], [54, 0, 38, 19, 46, 42, 56, 60], [77, 30, 24, 2, 3, 94, 98, 13]];
    var bias[DOUT] = [40, 72, 19, 95];

    signal input in[n][DIN];
    signal output out[n][DOUT];

    signal _sums[n][DOUT][DIN+1];
    for (var i = 0; i < n; i++) {
        for (var k = 0; k < DOUT; k++) {
            _sums[i][k][0] <== 0;
            for (var j = 0; j < DIN; j++) {
                _sums[i][k][j+1] <== _sums[i][k][j] + weight[k][j] * in[i][j];
            }
            out[i][k] <== _sums[i][k][DIN] + bias[k];
        }
    }
}

template mlp_small_fc3 (n) {
    var DIN = 4;
    var DOUT = 1;
    var weight[DOUT][DIN] = [[72, 26, 66, 52]];
    var bias[DOUT] = [67];

    signal input in[n][DIN];
    signal output out[n][DOUT];

    signal _sums[n][DOUT][DIN+1];
    for (var i = 0; i < n; i++) {
        for (var k = 0; k < DOUT; k++) {
            _sums[i][k][0] <== 0;
            for (var j = 0; j < DIN; j++) {
                _sums[i][k][j+1] <== _sums[i][k][j] + weight[k][j] * in[i][j];
            }
            out[i][k] <== _sums[i][k][DIN] + bias[k];
        }
    }
}

template mlp_small (n) {
    signal input in[n][16];
    signal output out[n][1];

    component _mlp_small_fc1 = mlp_small_fc1(n);
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < 16; j++) {
            _mlp_small_fc1.in[i][j] <== in[i][j];
        }
    }

    component _mlp_small_relu2d1 = relu2d(n, 8);
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < 8; j++) {
            _mlp_small_relu2d1.in[i][j] <== _mlp_small_fc1.out[i][j];
        }
    }

    component _mlp_small_fc2 = mlp_small_fc2(n);
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < 8; j++) {
            _mlp_small_fc2.in[i][j] <== _mlp_small_relu2d1.out[i][j];
        }
    }

    component _mlp_small_relu2d2 = relu2d(n, 4);
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < 4; j++) {
            _mlp_small_relu2d2.in[i][j] <== _mlp_small_fc2.out[i][j];
        }
    }

    component _mlp_small_fc3 = mlp_small_fc3(n);
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < 4; j++) {
            _mlp_small_fc3.in[i][j] <== _mlp_small_relu2d2.out[i][j];
        }
    }

    for (var i = 0; i < n; i++) {
        for (var j = 0; j < 1; j++) {
            out[i][j] <== _mlp_small_fc3.out[i][j];
        }
    }

}