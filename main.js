
const coeffs = [], constants = [];
const N = 1000;
for(let i = 0; i < N; i += 1) {
    coeffs.push(Mat.rand(20, 20));
    constants.push(Mat.rand(20, 10));
}

// console.log("PLU test");
// const startPLU = Date.now();
// for(let i = 0; i < N; i += 1) {
//     const A = coeffs[i], B = constants[i];
//     const plu = Mat.PLU(A);
//     const x = Mat.solvePLU(...plu, B);
//     const val = Mat.allClose(Mat.mul(A, x), B);
//     if(!val) console.log(A, B, x);
// }
// const endPLU = Date.now();
// console.log(endPLU - startPLU + " ms");

// console.log("\nGauss-Jordan test");
// const startGauss = Date.now();
// for(let i = 0; i < N; i += 1) {
//     const A = coeffs[i], B = constants[i];
//     const x = Mat.solveGauss(A, B);
//     const val = Mat.allClose(Mat.mul(A, x), B);
//     if(!val) console.log(A, B, x);
// }
// const endGauss = Date.now();
// console.log(endGauss - startGauss + " ms");

console.log("\nQR test");
const startQR = Date.now();
for(let i = 0; i < N; i += 1) {
    const A = coeffs[i], B = constants[i];
    const [Q, R] = Mat.QR(A);
    if(!Q.colsOrthogonal()) console.log("O", Q);
    if(!Q.colsUnitary()) console.log("U", Q);
    const QR = Mat.mul(Q, R);
    if(!Mat.allClose(QR, A)) console.log(Q, R, QR, A);
}
const endQR = Date.now();
console.log(endQR - startQR + " ms");
