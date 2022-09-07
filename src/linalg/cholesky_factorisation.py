import math
from vector import Vector
from matrix import Matrix

def cholesky(A: Matrix, b: Vector) -> Matrix:
    '''Applies Cholesky factorisation to decompose a symmetric positive-definite matrix A = (R^T)(R),
    where R is upper triangular
    
    A: Matrix
        Matrix object to be decomposed

    b: Vector

    return: Matrix
        Returns nxn matrix R s.t A = (R^T)(R), where R is upper triangular
    '''
    n, m = A.shape
    if n != m:
        raise Exception(f"Matrix must be square, received {n,m}")
    
    R = Matrix((n,n))
    for k in range(n):
        if A.get(k,k) < 0: 
            return R
        else:
            R_kk = math.sqrt(A.get(k,k)) 
            R.set(k,k, R_kk)
            
            if k == n-1:
                # no more submatrices, return R
                break
            
            u_T = (1/R_kk) * A.submatrix(k, (k+1,n))
            u_T = Matrix([u_T]) ## view as row matrix
            uu_T = u_T.transpose().matmul(u_T)
            R.set_submatrix(k, (k+1,n), u_T)

            new_principal_submat = A.submatrix((k+1,n), (k+1,n)) - uu_T

            A.set_submatrix((k+1,n), (k+1,n), new_principal_submat)
    
    return lu_solve(R.transpose(), R, b)
    
def lu_solve(L: Matrix, U: Matrix, b: Vector):
    '''General method to solve LUx = b
    
    Idea:
        Ax = b
        LUx = b
        Solve LY = b
        Solve Ux = Y
    '''
    rows = U.shape[0]

    # Solve LY = b: since L is lower triangular, this can be done via backsub in reverse order
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
    A = Matrix([[4,-2,2],[-2,2,-4],[2,-4,11]])
    b = Vector([0,-4,9])
    R = cholesky(A, b)
    print(R)