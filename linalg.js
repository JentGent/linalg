/**
 * Basic Matrix class linear algebra implementation libary
 */
class Mat {
    /**
     * Create a matrix
     * @param {number[]} arr - Nested array; `arr[n][m]` is the value at the `n`th row, `m`th column. If `arr` is not nested, `Mat` will turn it into a matrix with `N` rows and 1 column, where `N` is the number of elements in `arr`
     */
    constructor(arr) {
        if(!Array.isArray(arr)) throw new Error("Matrix must be a array");
        if(!Array.isArray(arr[0])) {
            for(let i = 0; i < arr.length; i += 1)
                arr[i] = [arr[i]];
        }
        this.arr = arr;
        this.rows = arr.length;
        this.cols = arr[0].length;
    }

    get square() { return this.rows === this.cols; }

    /**
     * Calculates whether or not the columns of the matrix are orthogonal (NOT unitary)
     * @param {number} tol - How close the dot product must be to 0
     * @returns {boolean} `true` if orthogonal
     */
    colsOrthogonal(tol = 0.00001) {
        for(let col1 = 0; col1 < this.cols - 1; col1 += 1) {
            for(let col2 = col1 + 1; col2 < this.cols; col2 += 1) {
                let dotProduct = 0;
                for(let row = 0; row < this.rows; row += 1)
                    dotProduct += this.arr[row][col1] * this.arr[row][col2];
                if(dotProduct > tol || dotProduct < -tol) return false;
            }
        }
        return true;
    }

    /**
     * Calculates whether or not the magnitudes of all the columns are 1
     * @param {number} tol - How close the dot product must be to 1
     * @returns {boolean} `true` if unitary
     */
    colsUnitary(tol = 0.00001) {
        for(let col = 0; col < this.cols; col += 1) {
            let dotProduct = 0;
            for(let row = 0; row < this.rows; row += 1) {
                dotProduct += this.arr[row][col] * this.arr[row][col];
                if(dotProduct > 1 + tol) return false;
            }
            if(dotProduct < 1 - tol) return false;
        }
        return true;
    }

    /**
     * Adds another matrix in place
     * @param {Mat} m - Addend
     * @returns {Mat} The matrix after adding
     */
    add(m) {
        if(!(m instanceof Mat)) throw new TypeError("Argument must be a matrix");
        if(this.rows !== m.rows || this.cols !== m.cols) throw new Error("Matrices must have same shape");
        for(let row = 0; row < this.rows; row += 1) {
            for(let col = 0; col < this.cols; col += 1)
                this.arr[row][col] += m.arr[row][col];
        }
        return this;
    }

    /**
     * Subtracts another matrix in place
     * @param {Mat} m - Subtrahend
     * @returns {Mat} The matrix after subtracting
     */
    sub(m) {
        if(!(m instanceof Mat)) throw new TypeError("Argument must be a matrix");
        if(this.rows !== m.rows || this.cols !== m.cols) throw new Error("Matrices must have same shape");
        for(let row = 0; row < this.rows; row += 1) {
            for(let col = 0; col < this.cols; col += 1)
                this.arr[row][col] -= m.arr[row][col];
        }
        return this;
    }

    /**
     * Multiplies all the values in the matrix by a number
     * @param {Mat} s - Scalar
     * @returns {Mat} The matrix after scaling
     */
    scale(s) {
        for(let row = 0; row < this.rows; row += 1) {
            for(let col = 0; col < this.cols; col += 1)
                this.arr[row][col] *= s;
        }
        return this;
    }

    /**
     * Swaps two rows/columns in place
     * @param {number} i - Index of row/column to swap
     * @param {number} j - Index of row/column to swap
     * @param {0 | 1} axis - Swap rows (0) or columns (1) (default `0`)
     * @returns {Mat} The matrix after swapping
     */
    swap(i, j, axis=0) {
        if(axis === 0) {
            if(this.arr[i] === undefined) throw new Error("Row '" + i + "' is undefined");
            if(this.arr[j] === undefined) throw new Error("Column '" + j + "' is undefined");
            [this.arr[i], this.arr[j]] = [this.arr[j], this.arr[i]];
            return this;
        }
        if(axis === 1) {
            for(let row = 0; row < this.rows; row += 1)
                [this.arr[row][i], this.arr[row][j]] = [this.arr[row][j], this.arr[row][i]];
            return this;
        }
        throw new Error("Axis must be 0 or 1");
    }
    
    /**
     * Converts a matrix in place into its reduced row-echelon form using Gauss-Jordan elimination with partial pivoting
     * @param {number} zero - Optionally, the threshold for stopping division by zero
     * @returns {Mat} The matrix in reduced row-echelon form
     */
    reduce(zero = 0) {
        let row = 0, col = 0;
        while(row < this.rows && col < this.cols) {
            // Find `maxRow` with highest value in `col`
            let maxRow = row, pivot = Math.abs(this.arr[row][col]);
            for(let i = maxRow + 1; i < this.rows; i += 1) {
                const n = Math.abs(this.arr[i][col]);
                if(n > pivot) {
                    maxRow = i;
                    pivot = n;
                }
            }
            if(pivot <= zero) {
                col += 1; // Skip `col` if all 0s
                continue;
            }
            if(row !== maxRow) [this.arr[maxRow], this.arr[row]] = [this.arr[row], this.arr[maxRow]]; // Move `maxRow` up
            // Normalize
            if(this.arr[row][col] !== 1) {
                const f = 1 / this.arr[row][col];
                this.arr[row][col] = 1;
                for(let i = col + 1; i < this.cols; i += 1)
                    this.arr[row][i] *= f;
            }
            // Cancel out all other rows
            for(let i = 0; i < this.rows; i += 1) {
                if(i === row) continue;
                const f = this.arr[i][col];
                this.arr[i][col] = 0;
                if(f)
                    for(let j = col + 1; j < this.cols; j += 1)
                        this.arr[i][j] -= this.arr[row][j] * f;
            }
            row += 1;
            col += 1;
        }
        return this;
    }

    /**
     * Extracts, in place, a submatrix from the original matrix, changing the original matrix shape
     * @param {number} rowStart - Row to start from (inclusive)
     * @param {number} rowEnd - Row to end at (exclusive)
     * @param {number} colStart - Column to start from (inclusive)
     * @param {number} colEnd - Column to end at (exclusive)
     * @returns {Mat} The submatrix
     */
    slice(rowStart = 0, rowEnd = this.rows, colStart = 0, colEnd = this.cols) {
        const newRows = rowEnd - rowStart,
              newCols = colEnd - colStart;
        if(rowStart !== 0 || colStart !== 0)
            for(let row = 0; row < newRows; row += 1)
                for(let col = 0; col < newCols; col += 1)
                    this.arr[row][col] = this.arr[row + rowStart][col + colStart];
        this.arr.length = newRows;
        if(newCols !== this.cols)
            for(let row = 0; row < newRows; row += 1)
                this.arr[row].length = newCols;
        this.rows = newRows;
        this.cols = newCols;
        return this;
    }

    /**
     * Replaces a subregion of the matrix with another matrix in place
     * @param {number} rowStart - Row to start from (inclusive)
     * @param {number} rowEnd - Row to end at (exclusive)
     * @param {number} colStart - Column to start from (inclusive)
     * @param {number} colEnd - Column to end at (exclusive)
     * @param {Mat} m - Matrix to replace the subregion with
     * @returns {Mat} The matrix after replacing the region
     */
    splice(rowStart = 0, rowEnd = this.rows, colStart = 0, colEnd = this.cols, m) {
        if(m.rows !== rowEnd - rowStart || m.cols !== colEnd - colStart) throw new RangeError("Matrix to replace region with must have the same shape as the region");
        if(!(m instanceof Mat)) throw new TypeError("Fifth argument must be a matrix");
        for(let row = rowStart; row < rowEnd; row += 1)
            for(let col = colStart; col < colEnd; col += 1)
                this.arr[row][col] = m.arr[row - rowStart][col - colStart];
        return this;
    }

    /**
     * Creates a new matrix with the same shape and values
     * @param {Mat} m - The matrix to copy
     * @returns {Mat} The duplicate matrix
     */
    static copy(m) {
        return new Mat(m.arr.map(a => [].concat(a)));
    }

    /**
     * Creates a new matrix by solving `Ax=B` for `x` with forward substitution for a lower triangular matrix `A`
     * @param {Mat} coeffs - An MxM lower triangular matrix `A`
     * @param {Mat} constants - An MxN matrix `B`
     * @returns {Mat} An MxN matrix `x` such that `Ax=B`
     */
    static forward(coeffs, constants) {
        if(!coeffs.square) throw new Error("Coefficient matrix must be square");
        if(coeffs.rows !== constants.rows) throw new Error("Mismatched matrix shapes; coefficients and constants must have the same number of rows");
        const newRows = [];
        for(let row = 0; row < coeffs.cols; row += 1) {
            const newRow = [];
            let f = coeffs.arr[row][row];
            f = f ? 1 / f : 0;
            for(let col = 0; col < constants.cols; col += 1) {
                if(f) {
                    let v = constants.arr[row][col];
                    for(let i = 0; i < row; i += 1)
                        v -= coeffs.arr[row][i] * newRows[i][col];
                    newRow[col] = v * f;
                }
                else newRow[col] = 0;
            }
            newRows[row] = newRow;
        }
        return new Mat(newRows);
    }

    /**
     * Creates a new matrix by solving `Ax=B` for `x` with back substitution for an upper triangular matrix `A`
     * @param {Mat} coeffs - An MxM upper triangular matrix `A`
     * @param {Mat} constants - An MxN matrix `B`
     * @returns {Mat} An MxN matrix `x` such that `Ax=B`
     */
    static backward(coeffs, constants) {
        if(!coeffs.square) throw new Error("Coefficient matrix must be square");
        if(coeffs.rows !== constants.rows) throw new Error("Mismatched matrix shapes; coefficients and constants must have the same number of rows");
        const newRows = [];
        for(let row = coeffs.cols - 1; row >= 0; row -= 1) {
            const newRow = [];
            let f = coeffs.arr[row][row];
            f = f ? 1 / f : 0;
            for(let col = 0; col < constants.cols; col += 1) {
                if(f) {
                    let v = constants.arr[row][col];
                    for(let i = coeffs.rows - 1; i > row; i -= 1)
                        v -= coeffs.arr[row][i] * newRows[i][col];
                    newRow[col] = v * f;
                }
                else newRol[col] = 0;
            }
            newRows[row] = newRow;
        }
        return new Mat(newRows);
    }

    /**
     * Creates a row permutation array and a new matrix by calculating the LU decomposition of the given matrix with Gaussian elimination and partial pivoting. The returned matrix is the "combination" of L and U; the upper triangle and the diagonal of the matrix have the values of U, while the lower half has the values of L
     * @param {Mat} m - The matrix to perform LU decomposition on
     * @param {number} zero - Optionally, the threshold for stopping division by zero
     * @returns {[number[], Mat]} The permutation array, and the L+U matrix
     * @example
     * ```
     * plu = Mat.PLU(A)
     * x = Mat.solvePLU(...plu, B)
     * Mat.mul(A, x) == B
     * ```
     */
    static PLU(m, zero = 0) {
        const P = [];
        for(let i = 0; i < m.rows; i += 1)
            P[i] = i;
        const LU = m.arr.map(x => [].concat(x));
        
        let row = 0, col = 0;
        while(row < m.rows && col < m.cols) {
            // Find `maxRow` with highest value in `col`
            let maxRow = row, pivot = Math.abs(LU[row][col]);
            for(let i = maxRow + 1; i < m.rows; i += 1) {
                const n = Math.abs(LU[i][col]);
                if(n > pivot) {
                    maxRow = i;
                    pivot = n;
                }
            }
            if(pivot <= zero) {
                col += 1; // Skip `col` if all 0s
                continue;
            }
            if(row !== maxRow) {
                [LU[maxRow], LU[row]] = [LU[row], LU[maxRow]]; // Move `maxRow` up
                [P[maxRow], P[row]] = [P[row], P[maxRow]];
            }
            let g = LU[row][col];
            if(g) {
                g = 1 / g;
                // Cancel out rows below
                for(let i = row + 1; i < m.rows; i += 1) {
                    if(i === row) continue;
                    const f = LU[i][col] * g;
                    LU[i][col] = f; // Lower matrix
                    if(f)
                        for(let j = col + 1; j < m.cols; j += 1)
                            LU[i][j] -= LU[row][j] * f;
                }
            }
            row += 1;
            col += 1;
        }
        return [P, new Mat(LU)];
    }

    /**
     * Creates two new matrices by separating a combined LU matrix (result of `Mat.PLU()`) into a lower and upper triangular matrix
     * @param {Mat} LU - The combined L+U matrix
     * @param {0 | 1} uni - Optionally, specifies whether the lower (0) or upper (1) matrix is unitriangular
     * @returns {Array<Mat, Mat>} The separated L and U matrices
     */
    static separateLU(LU, uni = 0) {
        const L = [], U = [];
        for(let row = 0; row < LU.rows; row += 1) {
            L[row] = [];
            U[row] = [];
            for(let col = 0; col < LU.cols; col += 1) {
                if(col < row) {
                    L[row][col] = LU.arr[row][col];
                    U[row][col] = 0;
                }
                else if(col === row) {
                    L[row][col] = uni ? LU.arr[row][col] : 1;
                    U[row][col] = uni ? 1 : LU.arr[row][col];
                }
                else {
                    L[row][col] = 0;
                    U[row][col] = LU.arr[row][col];
                }
            }
        }
        return [new Mat(L), new Mat(U)];
    }

    /**
     * Creates a new matrix by permuting the rows/columns of a matrix according to a permutation array
     * @param {Mat} m - The matrix to permute
     * @param {0 | number[]} pRows - Row permutation array (or `0` if no permutation)
     * @param {0 | number[]} pCols - Column permutation array (or `0` if no permutation)
     * @returns {Mat} The permuted matrix
     */
    static permute(m, pRows, pCols = 0) {
        const newRows = [];
        if(pCols === 0) {
            for(let i = 0; i < pRows.length; i += 1)
                newRows[i] = m.arr[pRows[i]].slice();
        }
        else if(pRows === 0) {
            for(let row = 0; row < m.rows; row += 1) {
                const newRow = [];
                for(let j = 0; j < pCols.length; j += 1)
                    newRow[j] = m.arr[row][pCols[j]];
                newRows[row] = newRow;
            }
        }
        else {
            for(let i = 0; i < pRows.length; i += 1) {
                const newRow = [];
                for(let j = 0; j < pCols.length; j += 1)
                    newRow[j] = m.arr[pRows[i]][pCols[j]];
                newRows[i] = newRow;
            }
        }
        return new Mat(newRows);
    }

    /**
     * Creates a new matrix by solving `LUx=PB` for `x`
     * @param {number[]} P - An array of M numbers representing the indices of the rows after pivoting in the LU decomposition
     * @param {Mat} LU - An MxM matrix `A` that is the result of the LU decomposition
     * @param {Mat} constants - An MxN matrix `B`
     * @returns {Mat} The solution `x` for `Ax=B`
     * @example
     * ```
     * plu = Mat.PLU(A)
     * x = Mat.solvePLU(...plu, B)
     * Mat.mul(A, x) == B
     * ```
     */
    static solvePLU(P, LU, constants) {
        if(!(LU instanceof Mat)) throw new TypeError("Second argument must be a matrix");
        if(!(constants instanceof Mat)) throw new TypeError("Third argument must be a matrix");
        if(!LU.square) throw new Error("Coefficient matrix must be square");
        if(LU.rows !== constants.rows) throw new Error("Mismatched matrix shapes; coefficients and constants must have the same number of rows");

        /** Separating is wasteful */
        // const [L, U] = Mat.separateLU(LU);
        // const Ux = Mat.forward(L, Mat.permute(constants, P));
        // const x = Mat.backward(U, Ux);
        // return x;

        // Forward: Y=Ux; solve LY=B for Y
        const Ux = [];
        for(let row = 0; row < LU.cols; row += 1) {
            const newRow = [];
            // The diagonals in L are 1
            for(let col = 0; col < constants.cols; col += 1) {
                let v = constants.arr[P[row]][col]; // Permuted constant
                for(let i = 0; i < row; i += 1)
                    v -= LU.arr[row][i] * Ux[i][col];
                newRow[col] = v;
            }
            Ux[row] = newRow;
        }

        // Backward: Solve Ux=Y for x
        const x = [];
        for(let row = LU.cols - 1; row >= 0; row -= 1) {
            const newRow = [];
            let f = LU.arr[row][row];
            f = f ? 1 / f : 0;
            for(let col = 0; col < constants.cols; col += 1) {
                if(f) {
                    let v = Ux[row][col];
                    for(let i = LU.rows - 1; i > row; i -= 1)
                        v -= LU.arr[row][i] * x[i][col];
                    newRow[col] = v * f;
                }
                else newRow[col] = 0;
            }
            x[row] = newRow;
        }
        
        return new Mat(x);
    }

    /**
     * Creates a new matrix by solving a system of linear equations `Ax=B` with Gauss-Jordan elimination
     * @param {Mat} coeffs - An AxB matrix of coefficients
     * @param {Mat} constants - An AxC matrix of constants
     * @returns {Mat} - A BxC matrix solution `x`
     */
    static solveGauss(coeffs, constants) {
        if(!coeffs.square) throw new Error("Coefficient matrix must be square");
        if(coeffs.rows !== constants.rows) throw new Error("Mismatched matrix shapes; coefficients and constants must have the same number of rows")
        const augmented = Mat.concat(coeffs, constants); // Ax(B+C)
        const reduced = augmented.reduce();
        return reduced.slice(0, coeffs.cols, reduced.cols - constants.cols);
    }

    static householder(normal) {
        if(!(normal instanceof Mat)) throw new TypeError("Argument must be a matrix");
        return Mat.identity(normal.rows, normal.rows).sub(Mat.mul(normal, Mat.T(normal)).scale(2));
    }

    static QR(m) {
        let QT = Mat.identity(m.rows), R = Mat.copy(m);
        for(let col = 0; col < Math.min(m.rows, m.cols); col += 1) {
            let r = 0;
            for(let row = col; row < m.rows; row += 1)
                r += R.arr[row][col] * R.arr[row][col];
            r = Math.sqrt(r);
            const target = [...Array(m.rows)].fill(0);
            for(let row = 0; row < col; row += 1)
                target[row] = R.arr[row][col];
            target[col] = r;
            let tr = 0;
            for(let row = 0; row < m.rows; row += 1) {
                target[row] -= R.arr[row][col];
                tr += target[row] * target[row];
            }
            const reflect = Mat.householder(new Mat(target).scale(1 / Math.sqrt(tr)));
            R = Mat.mul(reflect, R);
            R.arr[col][col] = r;
            for(let row = col + 1; row < m.rows; row += 1)
                R.arr[row][col] = 0;
            QT = Mat.mul(reflect, QT);
        }
        return [Mat.T(QT), R];
    }

    /**
     * Returns `true` if the following is element-wise true for all values in `m1` and `m2`: `|a-b|<=atol+rtol*|b|`
     * 
     * Otherwise, returns `false`
     * @param {Mat} m1 - Matrix
     * @param {Mat} m2 - Matrix to compare to
     * @param {number} rtol - Relative tolerance (default `1e-05`)
     * @param {number} atol - Absolute tolerance (default `1e-08`)
     * @returns {boolean} `true` if all values of the two matrices are equal within tolerance; otherwise, `false`
     */
    static allClose(m1, m2, rtol = 1e-05, atol = 1e-08) {
        if(!(m1 instanceof Mat) || !(m2 instanceof Mat)) throw new TypeError("Arguments must be matrices");
        if(m1.cols !== m2.cols || m1.rows !== m2.rows) throw new Error("Matrices must have same shape");
        for(let i = 0; i < m1.rows; i += 1) {
            for(let j = 0; j < m2.rows; j += 1) {
                if(Math.abs(m1.arr[i][j] - m2.arr[i][j]) > atol + rtol * Math.abs(m2.arr[i][j])) return false;
            }
        }
        return true;
    }

    /**
     * Creates a new matrix by extracting a submatrix from the original matrix
     * @param {Mat} m - The matrix to take a slice from
     * @param {number} rowStart - Row to start from (inclusive)
     * @param {number} rowEnd - Row to end at (exclusive)
     * @param {number} colStart - Column to start from (inclusive)
     * @param {number} colEnd - Column to end at (exclusive)
     * @returns {Mat} The submatrix
     */
    static slice(m, rowStart = 0, rowEnd = m.rows, colStart = 0, colEnd = m.cols) {
        const newRows = [];
        for(let row = rowStart; row < rowEnd; row += 1) {
            const newRow = [];
            for(let col = colStart; col < colEnd; col += 1)
                newRow[col - colStart] = m.arr[row][col];
            newRows[row - rowStart] = newRow;
        }
        return new Mat(newRows);
    }

    /**
     * Creates a new matrix by concatenating two matrices `m1` and `m2` along the `axis`
     * @param {Mat} m1 - An AxB matrix
     * @param {Mat} m2 - If `axis` is 0, a CxB matrix; if `axis` is 1, an AxC matrix
     * @param {0 | 1} axis - Concatenate rows (0) or columns (1)
     * @returns {Mat} If `axis` is 0, an (A+C)xB matrix; if `axis` is 1, an Ax(B+C) matrix
     */
    static concat(m1, m2, axis=1) {
        if(!(m1 instanceof Mat)) throw new TypeError("First argument must be a matrix");
        if(!(m2 instanceof Mat)) throw new TypeError("Second argument must be a matrix");
        if(axis === 0) {
            if(m1.cols !== m2.cols) throw new Error("Mismatched matrix shapes");
            return new Mat(m1.arr.concat(m2.arr));
        }
        if(axis === 1) {
            if(m1.rows !== m2.rows) throw new Error("Mismatched matrix shapes");
            const newRows = [];
            for(let row = 0; row < m1.rows; row += 1) newRows[row] = m1.arr[row].concat(m2.arr[row]);
            return new Mat(newRows);
        }
        throw new Error("Axis must be 0 or 1");
    }

    /**
     * Creates a new matrix by filling a matrix of the given dimensions with random values between a given range
     * @param {number} rows - Number of rows
     * @param {number} cols - Number of columns
     * @param {number} min - Minimum value (defaults `-1`)
     * @param {number} max - Maximum value (defaults `1`)
     * @returns {Mat} An `m`x`n` matrix of random numbers between `min` and `max`
     */
    static rand(rows, cols = rows, min = -1, max = 1) {
        return new Mat([...Array(rows)].map(row => [...Array(cols)].map(col => Math.random() * (max - min) + min)));
    }

    /**
     * Creates a new matrix by filling a matrix of the given dimensions with `1`s along the principal diagonal and `0`s everywhere else
     * @param {number} rows - Dimensions of the matrix
     * @param {number} cols - Optionally, some other number of columns; if different from `m`, this won't be an identity matrix
     * @returns {Mat} Matrix with `1`s along diagonal and `0`s everywhere else
     */
    static identity(rows, cols = rows) {
        return new Mat([...Array(rows)].map((row, i) => [...Array(cols)].map((col, j) => i === j ? 1 : 0)));
    }

    /**
     * Creates a new matrix by filling a matrix of the given dimensions with a given value
     * @param {number} rows - Number of rows
     * @param {number} cols - Number of columns
     * @param {number} value - Value to fill the matrix with
     * @returns {Mat} Matrix filled with the given value
     */
    static fill(rows, cols = rows, value = 0) {
        return new Mat([...Array(rows)].map(row => Array(cols).fill(value)));
    }

    /**
     * Creates a new matrix by multiplying two matrices, or scaling a matrix by a number
     * @param {Mat} m1 - AxB matrix
     * @param {Mat | number} m2 - BxC matrix
     * @returns {Mat} AxC matrix product
     */
    static mul(m1, m2) {
        if(!(m1 instanceof Mat)) throw new TypeError("First argument must be a matrix");
        if(typeof m2 === "number") {
            const newRows = [];
            for(let row = 0; row < m1.rows; row += 1) {
                const newRow = [];
                for(let col = 0; col < m1.cols; col += 1)
                    newRow[col] = m1.arr[row][col] * m2;
                newRows[row] = newRow;
            }
            return new Mat(newRows);
        }
        if(!(m2 instanceof Mat)) throw new TypeError("Second argument must be a matrix or number");
        if(m1.cols !== m2.rows) throw new Error("Mismatched matrix shapes");
        const newRows = [];
        for(let row = 0; row < m1.rows; row += 1) {
            const newRow = [];
            for(let col = 0; col < m2.cols; col += 1) {
                let v = 0;
                for(let k = 0; k < m1.cols; k += 1)
                    v += m1.arr[row][k] * m2.arr[k][col];
                newRow[col] = v;
            }
            newRows[row] = newRow;
        }
        return new Mat(newRows);
    }

    /**
     * Creates a new matrix by summing two AxB matrices
     * @param {Mat} m1 - Addend
     * @param {Mat} m2 - Addend
     * @returns {Mat} Sum
     */
    static add(m1, m2) {
        if(!(m1 instanceof Mat) || !(m2 instanceof Mat)) throw new TypeError("Arguments must be matrices");
        if(m1.rows !== m2.rows || m1.cols !== m2.cols) throw new Error("Matrices must have same shape");
        const newRows = [];
        for(let row = 0; row < m1.rows; row += 1) {
            const newRow = [];
            for(let col = 0; col < m1.cols; col += 1)
                newRow[col] = m1.arr[row][col] + m2.arr[row][col];
        }
        return new Mat(newRows);
    }

    /**
     * Creates a new matrix by subtracting one AxB matrix from another
     * @param {Mat} m1 - Minuend
     * @param {Mat} m2 - Subtrahend
     * @returns {Mat} Difference
     */
    static sub(m1, m2) {
        if(!(m1 instanceof Mat) || !(m2 instanceof Mat)) throw new TypeError("Arguments must be matrices");
        if(m1.rows !== m2.rows || m1.cols !== m2.cols) throw new Error("Matrices must have same shape");
        const newRows = [];
        for(let row = 0; row < m1.rows; row += 1) {
            const newRow = [];
            for(let col = 0; col < m1.cols; col += 1)
                newRow[col] = m1.arr[row][col] - m2.arr[row][col];
            newRows[row] = newRow;
        }
        return new Mat(newRows);
    }

    /**
     * Creates a new matrix by inverting the given matrix with PLU decomposition
     * 
     * Matrix inversion as an intermediate step of a method can often be replaced with more efficient processes!
     * @param {Mat} m - The matix to invert
     * @returns {Mat} The inverse
     */
    static invert(m) {
        return Mat.solvePLU(...Mat.PLU(m), Mat.identity(m.rows));
    }

    /**
     * Creates a new BxA matrix by transposing an AxB matrix
     * @param {Mat} m
     * @returns {Mat} Transposed matrix
     */
    static T(m) {
        if(!(m instanceof Mat)) throw new TypeError("Argument must be a matrix");
        const newRows = [];
        for(let row = 0; row < m.cols; row += 1) {
            const newRow = [];
            for(let col = 0; col < m.rows; col += 1)
                newRow[col] = m.arr[col][row];
            newRows[row] = newRow;
        }
        return new Mat(newRows);
    }

    /**
     * Creates a new matrix by solving for the least squares matrix solution of an overdetermined system of linear equations `Ax=B`, i.e., minimizing `dist(Ax, B)`
     * @param {Mat} coeffs - An AxB matrix of coefficients
     * @param {Mat} constants - An AxC matrix of constants
     * @returns {Mat} - A BxC matrix `x` that minimizes `dist(Ax, B)`
     */
    static lstsq(coeffs, constants) {
        if(!(coeffs instanceof Mat) || !(constants instanceof Mat)) throw new TypeError("Arguments must be matrices");
        if(coeffs.rows < coeffs.cols) throw new Error("Coefficients matrix must have at least as many rows as columns");
        if(coeffs.rows !== constants.rows) throw new Error("Mismatched matrix shapes; coefficients and constants must have the same number of rows");
        const T = Mat.T(coeffs); // BxA
        // T * coeffs    = BxA * AxB = BxB
        // T * constants = BxA * AxC = BxC
        const augmented = Mat.concat(Mat.mul(T, coeffs), Mat.mul(T, constants)); // Bx(B+C)
        const reduced = augmented.reduce();
        return reduced.slice(0, reduced.rows, coeffs.cols);
    }
}