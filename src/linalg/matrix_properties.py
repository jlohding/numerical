'''Static methods to check if matrix satisfies some property
'''
from matrix import Matrix

class MatrixClassifier:
    def is_triangular(self, A: Matrix) -> bool:
        return self.is_upper_triangular() or self.is_lower_triangular()

    def is_lower_triangular(self, A: Matrix) -> bool:
        rows, cols = A.shape
        for r in range(rows):
            for c in range(r+1, cols):
                if A.get(r,c) != 0:
                    return False
        return True

    def is_upper_triangular(self, A: Matrix) -> bool:
        rows, cols = A.shape
        for r in range(rows):
            for c in range(0,r):
                if A.get(r,c) != 0:
                    return False
        return True

    def is_strictly_diagonally_dominant(self, A: Matrix) -> bool:
        if not self.is_square(A):
            return False
        else:
            order = A.shape[0]
            for i in range(order):
                A_ii = A.get(i,i)
                for c in range(order):
                    if i == c:
                        continue
                    if abs(A.get(i,c)) > A_ii:
                        return False
            return True

    def is_square(self, A: Matrix) -> bool:
        return A.shape[0] == A.shape[1]