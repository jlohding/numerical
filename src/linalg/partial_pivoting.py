from vector import Vector
from matrix import Matrix

def partial_pivoting(A: Matrix, b: Vector):
    '''Solves for Ax = b via partial pivoting gaussian elimination and backsubstitution

    Idea/Algorithm:
        for column c : 0 -> (numCols-1):
            *** Swap rows s.t a_{c,c} > a_{i, c} for all i (partial pivoting) ***
            use elementary row operations to zero all entries below the main diagonal entries: { a_{i,j} | a_{c+i, c} for i = 1,2,3,...,n }
                - For example, to make entry a_{i,j} zero, we need to do the operation: row(i) - ( a_{i,j} / a_{i-1,j} ) * row(i-1) )

        Then, we do backsubstitution which is:
        for row r : numRows-1 -> -1:
            X_{r} = ( b_{1} - a_{1,c}*x_2 - a_{2,c}*x_3 - ... - a_{numCols-1,c}*x_num_cols ) / a_{r,r}  

    A: Matrix
        n x p matrix object
    b: Vector
        1 x p vector object

    return: Vector
        returns 1 x p Vector object
    '''
    A = A.copy()
    rows, cols = A.shape

    # Gaussian Elimination Step
    for c in range(0, cols):
        # get the argmax of the pivots in this column, then swap for both A and b  
        max_pivot = c
        for i in range(c+1, rows):
            if A.get(i,c) > A.get(max_pivot,c):
                max_pivot = i 
        A.swap_rows(c, max_pivot)
        b.swap(c, max_pivot)

        for r in range(c+1, cols):        
            multiplier = A.get(r,c) / A.get(c,c)
            for i in range(c, cols):
                val = A.get(r,i)
                new_entry = val - (multiplier * A.get(c,i))
                A.set(r, i, new_entry)
            b.set(r, b.get(r) - b.get(c)*multiplier) 

    # Back-substitution Step
    X = Vector(0, size=rows)

    for r in range(rows-1,-1,-1):
        for c in range(r+1, rows):
            new_b = b.get(r) - A.get(r,c)*X.get(c)
            b.set(r, new_b)
        X.set(r, b.get(r) / A.get(r,r))

    return X

if __name__ == "__main__":
    A = Matrix([[1,2,-1],[2,1,-2],[-3,1,1]])
    b = Vector([3,3,-6])
    X = partial_pivoting(A, b)
    print(X)