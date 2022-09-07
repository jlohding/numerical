from vector import Vector
from typing import List, Tuple

class Matrix:
    def __init__(self, values: List[List[float] or Vector] or Tuple[int, int]):
        '''Constructor for matrix object
        The constructor is overloaded to take in either values: List or Tuple
        where List will construct a pre-filled matrix with values, and 
        Tuple will construct an empty zero matrix of a specified shape

        values: List[List[float] or Vector] or Tuple[int, int]
            List[List[float] or Vector]: pre-populated matrix with values
                This should be a list of lists or vectors, and should have the shape (rows, columns)
                
            Tuple<int, int>: shape of empty matrix 
                Tuple in the form (rows, columns)
                A zero matrix with the size (values[0], values[1]) will be init
        '''    
        if not isinstance(values, (List, Tuple)):
            raise Exception(f"Wrong input type, must be either List or Tuple, received {type(values)}")

        if isinstance(values, tuple):
            # Empty matrix constructor
            rows, columns = values
            self.matrix = [Vector(0, size=columns) for _ in range(rows)]

        elif isinstance(values, List):
            # Pre-filled Matrix from Lists or Vectors
            if isinstance(values[0], Vector):
                self.matrix = values
            elif isinstance(values[0], List):
                # convert to Vectors
                self.matrix = [Vector(x) for x in values]
            else:
                raise Exception(f"Wrong input type: List must nest either List or Vector, received {type(values[0])}")
    
        self.shape = (len(self.matrix), len(self.matrix[0]))

    def get(self, i: int, j: int) -> float:
        '''Returns value of (i,j)-entry

        i: int
            row index
        j: int
            column index

        return: float
            value of (i,j)-entry of matrix
        '''
        return self.matrix[i].get(j)

    def set(self, i: int, j: int, val: float) -> 'Matrix':
        '''Sets value at (i,j)-entry

        i: int
            row index
        j: int
            column index
        '''
        self.matrix[i].set(j, val)
        return self
    
    def triu(self) -> 'Matrix':
        '''Returns the upper triangle of the matrix
        '''
        rows, columns = self.shape
        tri = Matrix((rows, columns))
        for i in range(rows):
            for j in range(i+1, columns):
                tri.set(i,j, self.get(i,j))
        return tri

    def tril(self) -> 'Matrix':
        '''Returns the lower triangle of the matrix
        '''
        rows, columns = self.shape
        tri = Matrix((rows, columns))
        for j in range(columns):
            for i in range(j+1, rows):
                tri.set(i,j, self.get(i,j))
        return tri

    def diagonal(self) -> 'Matrix':
        '''Returns the diagonal of the matrix
        '''
        rows, columns = self.shape
        diag = Matrix((rows, columns))
        for i in range(min(rows, columns)):
            diag.set(i,i, self.get(i,i))
        return diag

    def swap_rows(self, i: int, j: int) -> 'Matrix':
        '''Swaps rows i and j
        
        i, j: int
            row indices
        '''
        self.matrix[i], self.matrix[j] = self.matrix[j], self.matrix[i]
        return self

    def transpose(self, inplace: bool = False) -> 'Matrix':
        '''Returns Matrix object with transposed entries
        
        inplace: bool
            if True, will transpose this matrice's values in place and return self
            else, returns a new Matrix object with a different reference

        return: Matrix
            Matrix object A^T with entries A^T_{i,j} = A_{j,i}  
        '''
        transposed_values = list(list(x) for x in zip(*self.to_list()))
        if inplace == True:
            self = Matrix(transposed_values)
            return self
        else:
            return Matrix(transposed_values)

    def p_norm(self, p: float) -> float:
        '''Computes the matrix norm for matrix A

        p: float
            Options are p=1 and p = float("inf")
            When p=1, ||A|| = max{ abs column sum }
            When p=inf, ||A|| = max{ abs row sum } 
        '''
        if p not in [1, float("inf")]:
            raise NotImplementedError("Matrix norm for p != 1 and p != inf is not implemented")
        else:
            if p == 1:
                return max(vec.p_norm(1) for vec in self.transpose().matrix)
            else:
                return max(vec.p_norm(1) for vec in self.matrix)

    def matmul(self, __o: 'Matrix' or 'Vector') -> 'Matrix':
        '''Matrix Multiplication

        __o : Matrix or Vector
            Must have shape = (columns, j)

        n x m m x p

        return: Matrix or Vector
            Returns result of matrix multiplication
        '''
        res = Matrix((self.shape[0], __o.shape[1]))

        if isinstance(__o, Vector):
            for r, vec in enumerate(self.matrix):
                value = vec.dot(__o)
                res.set(r, 0, value)

        elif isinstance(__o, Matrix):
            for r, vec in enumerate(self.matrix):
                for c, o_vec in enumerate(__o.transpose().matrix):
                    value = vec.dot(o_vec)
                    res.set(r,c, value)
        return res

    def submatrix(self, rows: Tuple[int,int] or int, cols: Tuple[int,int] or int) -> 'Matrix' or 'Vector' or int:
        '''Returns submatrix A_(rows[0] to rows[1], cols[0] to cols[1])
        
        rows: Tuple[int, int] or int
            Left and right bound [a, b) for submatrix rows
            If type(rows) == int, select only one single row        

        cols: Tuple[int, int]
            Left and right bound [a, b) for submatrix cols 
            If type(cols) == int, select only one single col        

        return: Matrix or Vector or int
            Returns new object of type depending on shape of submatrix
        '''
        if isinstance(rows, int) and isinstance(cols, int):
            # single entry
            return self.get(rows, cols)

        elif isinstance(rows, int):
            return Vector([self.get(rows, c) for c in range(*cols)])

        elif isinstance(cols, int):
            return Vector([self.get(r, cols) for r in range(*rows)])
        
        else:
            vecs = [Vector([self.get(r,c) for c in range(*cols)]) for r in range(*rows)]
            return Matrix(vecs)

    def set_submatrix(self, rows: Tuple[int, int] or int, cols: Tuple[int, int] or int, submatrix: 'Matrix') -> 'Matrix':
        '''Sets a matrix's values into an equally-shaped submatrix at specified rows/cols
        For example: A = [[1,0,0],[0,1,0],[0,0,1]], B = [[5,5],[5,5]]
        A.set_submatrix((0,2),(0,2), B) returns Matrix([5,5,0],[5,5,0],[0,0,1])
        
        rows: Tuple[int, int] or int
            Left and right bound [a, b) for submatrix rows
            If type(rows) == int, select only one single row        

        cols: Tuple[int, int]
            Left and right bound [a, b) for submatrix cols 
            If type(cols) == int, select only one single col        

        submatrix: Matrix
            submatrix.shape must equal (rows[1]-rows[0], cols[1]-cols[0])

        return: Matrix
            Returns reference to self
        '''
        if isinstance(rows, int) and isinstance(cols, int):
            # single entry
            self.set(rows, cols, submatrix.get(rows,cols))

        elif isinstance(rows, int):
            for i, c in enumerate(range(*cols)):
                self.set(rows, c, submatrix.get(0,i))

        elif isinstance(cols, int):
            for i, r in enumerate(range(*rows)):
                self.set(r, cols, submatrix.get(i,0))
        
        else:
            if submatrix.shape != (rows[1]-rows[0], cols[1]-cols[0]):
                raise Exception(f"Submatrix shape {submatrix.shape} does not align with target")

            for i, r in enumerate(range(*rows)):
                for j, c in enumerate(range(*cols)):
                    self.set(r,c, submatrix.get(i,j))

        return self

    def to_vector(self) -> Vector:
        '''Converts to 1-dimensional column vector
        
        The shape of the matrix must be (n, 1)
        '''
        if self.shape[1] != 1:
            raise Exception("Matrix must have only one column to convert to vector")
        else:
            return Vector([v.get(0) for v in self.matrix])

    def to_list(self) -> List[List[float]]:
        '''Getter for list of list matrix
        '''
        return [vec.to_list() for vec in self.matrix]

    def copy(self) -> 'Matrix':
        '''Returns deepcopy of 2 levels
        '''
        return Matrix([vec.copy() for vec in self.to_list()])

    def __str__(self) -> str:
        '''Override __str__ to display matrix in 2d format
        '''
        mat = self.to_list()
        return "\n".join(str(row) for row in mat)      

    def __eq__(self, __o: object) -> bool:
        '''Override __eq__ to check elementwise equality
        '''
        if not isinstance(__o, Matrix):
            return False
        else:
            return self.to_list() == __o.to_list()

    def __add__(self, __o: 'Matrix') -> 'Matrix':
        '''Override __add__ for matrix elementwise addition
        '''
        if not isinstance(__o, (Vector,Matrix)):
            raise Exception("Cannot add Matrix with non-Vector or non-Matrix")
        
        if isinstance(__o, Vector):
            # convert to Matrix
            __o = Matrix(__o)

        if self.shape != __o.shape:
            raise Exception("Cannot add matrices of different length")
        else:
            res = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    a = self.get(i,j)
                    b = __o.get(i,j)
                    res.set(i,j, a+b)
            return res 

    def __sub__(self, __o: 'Vector') -> 'Vector':
        '''Override __sub__ for matrix elementwise subtraction
        '''
        if not isinstance(__o, (Vector,Matrix)):
            raise Exception("Cannot add Matrix with non-Vector or non-Matrix")
        
        if isinstance(__o, Vector):
            # convert to Matrix
            __o = Matrix(__o)

        if self.shape != __o.shape:
            raise Exception("Cannot add matrices of different length")
        else:
            res = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    a = self.get(i,j)
                    b = __o.get(i,j)
                    res.set(i,j, a-b)
            return res 

    def __mul__(self, k: float) -> 'Matrix':
        '''Override __mul__ for scalar multiplication
        '''
        if not isinstance(k, (int, float)):
            raise Exception("Must pass in scalar for scalar multiplication")
        else:
            return Matrix([vec*k for vec in self.matrix])

    def __rmul__(self, k: float) -> 'Matrix':
        '''Override __rmul__ for scalar multiplication (commutative)
        '''
        return self.__mul__(k)

if __name__ == "__main__":
    mat = Matrix([[1,2,3],[4,5,6],[7,8,9]])

    submat = mat.submatrix((0,3),1)

    print(submat, "\n")
    mat.set_submatrix((0,3), 1, submat)
    print(mat)
