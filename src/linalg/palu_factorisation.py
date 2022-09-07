from vector import Vector
from matrix import Matrix
from special_matrices import IdentityMatrix

def palu_factorisation(A: Matrix, b: Vector) -> Vector:
    '''Solves system of equations Ax = b via PA=LU Factorisation

    Idea/Algorithm:
        Decompose A into LU via Gaussian Elimination first, where L = lower triangular, U = upper triangular
            PALU idea:
                Maintain a permutation matrix P, keeping track of the cumulative permutations of the rows as we do partial pivoting
                In order to keep track of the multipliers, we can fit them into the eliminated entries, and the permutation matrix P will shift them to their correct positions
                at the PA = LU step

        After obtaining PA = LU,
            Ax = b <->
            PAx = Pb <->
            LUx = Pb
        Solve L(y) = Pb for (y)
        Solve Ux = y for x 

    A: Matrix
        Input matrix
    b: Vector
        1 x p vector

    return: Vector
        returns vector X, solution to the system
    '''
    A = A.copy()
    U = A.copy()
    rows, cols = A.shape
    P = IdentityMatrix((min(rows,cols), min(rows,cols)))

    # Gaussian Elimination Step to get U + Set multipliers to entries in L
    for c in range(0, cols-1):
        
        # partial pivot
        max_pivot = c
        for i in range(c+1, rows):
            if U.get(i,c) > U.get(max_pivot,c):
                max_pivot = i 
        U.swap_rows(c, max_pivot)
        P.swap_rows(c, max_pivot) # permutation matrix update

        for r in range(c+1, cols):        
            multiplier = U.get(r,c) / U.get(c,c)
            for i in range(c, cols):
                val = U.get(r,i)
                new_entry = val - (multiplier * U.get(c,i))
                if i == c:
                    # store the multiplier in the empty cells
                    U.set(r,i, multiplier)
                else:
                    U.set(r, i, new_entry)

    # split into LU
    L = U.tril()
    for i in range(min(rows, cols)):
        L.set(i,i,1)
    
    temp = U
    U = U.triu()
    for i in range(min(rows, cols)):
        U.set(i,i,temp.get(i,i))

    # Permute b:
    Pb = P.matmul(b).to_vector()

    # Solve LY = (Pb): since L is lower triangular, this can be done via backsub in reverse order
    Y = Vector(0, size=rows)

    for r in range(0,rows):
        for c in range(0, rows):
            new_Pb = Pb.get(r) - L.get(r,c)*Y.get(c)
            Pb.set(r, new_Pb)
        Y.set(r, Pb.get(r) / L.get(r,r))

    # Solve Ux = Y: since U is upper triangular, this is done via backsubstitution (normal order) 
    X = Vector(0, size=rows)
    
    for r in range(rows-1,-1,-1):
        for c in range(r+1, rows):
            new_b = Y.get(r) - U.get(r,c)*X.get(c)
            Y.set(r, new_b)
        X.set(r, Y.get(r) / U.get(r,r))
    
    return X
        
if __name__ == "__main__":
    A = Matrix([[2,1,5],[4,4,-4],[1,3,1]])
    b = Vector([5,0,6])
    print(palu_factorisation(A, b))