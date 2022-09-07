from vector import Vector
from matrix import Matrix

def lu_factorisation(A: Matrix, b: Vector) -> Vector:
    '''Solves system of equations Ax = b via LU Factorisation

    Idea/Algorithm:
        Decompose A into LU via Gaussian Elimination first, where L = lower triangular, U = upper triangular
        Then Ax = b <-> LUx = b:
            Get A = LU
            Solve Ly = b for y
            Then solve Ux = y for the solution x
            This will take one gaussian elimination O(n^3) and 2 back_substitutions O(n^2)

    A: Matrix
        Input matrix
    b: Vector
        1 x p vector

    return: Vector
        returns vector X, solution to the system
    '''
    U = A.copy()
    rows, cols = A.shape

    # init lower triangular matrix with diagonal = 1
    L = Matrix((rows, cols))
    for i in range(min(rows, cols)):
        L.set(i,i, 1)

    # Gaussian Elimination Step to get U + Set multipliers to entries in L
    for c in range(0, cols-1):
        for r in range(c+1, cols):        
            multiplier = U.get(r,c) / U.get(c,c)
            L.set(r, c, multiplier)
            for i in range(c, cols):
                val = U.get(r,i)
                new_entry = val - (multiplier * U.get(c,i))
                U.set(r, i, new_entry)

    # Solve LY = b: since L is lower triangular, this can be done via backsubstitution (in reverse order)
    Y = Vector(0, size=rows)

    for r in range(0,rows):
        for c in range(0, rows):
            new_b = b.get(r) - L.get(r,c)*Y.get(c)
            b.set(r, new_b)
        Y.set(r, b.get(r) / L.get(r,r))

    # Solve Ux = Y: since U is upper triangular, this is done via backsubstitution (normal order) 
    X = Vector(0, size=rows)
    
    for r in range(rows-1,-1,-1):
        for c in range(r+1, rows):
            new_b = Y.get(r) - U.get(r,c)*X.get(c)
            Y.set(r, new_b)
        X.set(r, Y.get(r) / U.get(r,r))
    
    return X
        
if __name__ == "__main__":
    A = Matrix([[1,2,-1],[2,1,-2],[-3,1,1]])
    b = Vector([3,3,-6])
    print(lu_factorisation(A, b))

