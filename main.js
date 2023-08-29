
console.log("PLU test");
const startPLU = Date.now();
for(let i = 0; i < 1000; i += 1) {
    const A = Mat.rand(20, 20),
          B = Mat.rand(20, 10);
    const plu = Mat.PLU(A);
    const x = Mat.solvePLU(...plu, B);
    console.log(Mat.allClose(Mat.mul(A, x), B));
}
const endPLU = Date.now();
console.log(endPLU - startPLU + " ms");

console.log("\nGauss-Jordan test");
const startGauss = Date.now();
for(let i = 0; i < 1000; i += 1) {
    const A = Mat.rand(20, 20),
          B = Mat.rand(20, 10);
    const x = Mat.solveGauss(A, B);
    console.log(Mat.allClose(Mat.mul(A, x), B));
}
const endGauss = Date.now();
console.log(endGauss - startGauss + " ms");
