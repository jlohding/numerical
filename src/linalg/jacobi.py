from vector import Vector
from matrix import Matrix
from special_matrices import IdentityMatrix
from matrix_properties import MatrixClassifier

import warnings

def jacobi(A: Matrix, b: Vector, max_iters=50) -> Vector:
    '''Faster Jacobi fixed-point iteration method by decomposing A into (D + L + U), the sum of a diagonal, tril and triu matrix
    
    Idea/Algorithm:
        Ax = b
        (D + L + U)x = b
        Dx = b - (L+U)x
        x = (D^-1) (b - (L+U)x)
        Then set x_0 = (0, 0, ..., 0)
        x_{k+1} = (D^-1).matmul(b-(L+U)x_{k}) for k = 0,1,2,3,..... until convergence or max_iters is hit

        The formal notation is:    
            x := (D^-1)(b - (L+U).(x))

    A: Matrix
        Square matrix
    b: Vector
        Vector to solve for
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
    converged = False
    iteration_count = 0

    while not converged:
        iteration_count += 1        
        new_x = D_inverse.matmul((b - (L+U).matmul(x).to_vector()))

        converged = (new_x == x) or iteration_count > max_iters
        x = new_x
    return x

def naive_jacobi(A: Matrix, b: Vector, max_iters=50) -> Vector:
    '''Solves Ax = b via Jacobi Fixed Point Iteration (FPI) method

    Idea/Algorithm:
        Given A = [v_1, v_2, ..., v_n]^T, where v_i is a row vector,
        Begin with an initial estimate for [b_1, ..., b_n], usually all zeroes
        Then solve algebraically the nth equation for the nth unknown, and obtain i equations in the matrix [u_1, u_2, ..., u_n]^T
        Sub [b_1, ..., b_n] into the above matrix to obtain a new set of estimates := [b_1, b_2, ..., b_n]
        Repeat until convergence condition is met

    Convergence conditions:
        If A is strictly diagonally dominant, then convergence to the unique solution is guaranteed
            The proof is by showing that the spectral radius p((D^-1)(L+U)) < 1 
       Otherwise, the Jacobi method may diverge and no solution will be found

    A: Matrix
        Linear system with n unknowns and at least n rows: this matrix must be square
    b: Vector
        Vector to solve for
    '''
    if not MatrixClassifier().is_square(A):
        raise Exception("Invalid input, A must be square matrix")
    if not MatrixClassifier().is_strictly_diagonally_dominant(A):
        warnings.warn("Matrix is not strictly diagonally dominant, system may not converge to unique solution")

    rows, cols = A.shape
    # begin with zero vector
    guess = Vector(values=0, size=len(b))
    converged = False
    iteration_count = 0

    while not converged:
        iteration_count += 1
        new_guess = Vector(values=0, size=len(b))
        for i in range(rows):
            new_b_i = b.get(i)
            for c in range(cols):
                if c != i:
                    new_b_i -= (A.get(i,c) * guess.get(c))
            new_b_i /= A.get(i,i)
            new_guess.set(i,new_b_i)
        
        converged = (guess == new_guess) or (iteration_count > max_iters)
        guess = new_guess
    
    return guess

if __name__ == "__main__":
    A = Matrix([[3,1],[1,2]])
    A2 = Matrix([[1,2],[3,1]]) # this will not converge
    b = Vector([5,5])
    
    X = jacobi(A,b)
    print(X)