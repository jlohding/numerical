from vector import Vector
from matrix import Matrix
from special_matrices import IdentityMatrix
from matrix_properties import MatrixClassifier

import warnings

def gauss_seidel(A: Matrix, b: Vector, max_iters: int = 50, sor: float = 1):
    '''Solves a square matrix A_nxn for system Ax = b using the Gauss-Seidel method

    Idea/Algorithm:
        The Gauss-Seidel Method (GSM) is a variation of the Jacobi method, where the entries in the approximation vector x 
        to be updated is updated immediately within the x_k -> x_{k+1} step, instead of waiting for the entire vector to be updated

        The matrix notation for the algorithm is:
            Ax = b
            (D + L + U)x = b
            x_{k+1} := (D^-1)(b - Ux_{k} - Lx_{k+1}) for k = 0,1,2....
    
    A: Matrix
        Square matrix, guaranteed convergence if A is strictly diagonally dominant

    b: Vector
        Vector to be solved for

    max_iters: int
        Maximum number of iterations without convergence before algorithm terminates

    sor: float
        default = 1, Successive Over-Relaxation (SOR) with w=1 is equivalent to the GSM
        Else, sets the SOR relaxation parameter w 
        Idea: Define each component in the guess vector x as a weighted average of the GSM and the previous guess,
        where the respective weights are (w) and (1-w)
    '''
    if not MatrixClassifier().is_square(A):
        raise Exception("Invalid input, A must be square matrix")
    if not MatrixClassifier().is_strictly_diagonally_dominant(A):
        warnings.warn("Matrix is not strictly diagonally dominant, system may not converge to unique solution")

    rows, cols = A.shape

    # Decompose A into (D + L + U)
    D = A.diagonal()
    L = A.tril()
    U = A.triu()

    D_inverse = D.copy()
    for i in range(D.shape[0]):
        D_inverse.set(i,i, 1/D.get(i,i))

    # initialise x = zero vector
    x = Vector(values=0, size=len(b))
    new_x = x.copy()
    converged = False
    iteration_count = 0

    if sor == 1: 
        # default GSM
        while not converged:
            iteration_count += 1        
            new_x = D_inverse.matmul((b - U.matmul(x).to_vector() - L.matmul(new_x).to_vector()))
            converged = (new_x == x) or iteration_count > max_iters
            x = new_x
        return x
    else:
        raise NotImplementedError("SOR is not implemented yet")

if __name__ == "__main__":
    A = Matrix([[3,1,-1],[2,4,1],[-1,2,5]])
    b = Vector([4,1,1])
    X = gauss_seidel(A,b, sor=1)
    print(X)