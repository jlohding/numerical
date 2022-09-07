from typing import Tuple
from matrix import Matrix

class IdentityMatrix(Matrix):
    def __init__(self, values: Tuple[int, int]):
        '''Constructor for identity matrix
        '''
        if values[0] != values[1]:
            raise Exception("Identity matrix must be square")

        super().__init__(values)
        for i in range(self.shape[0]):
            self.set(i,i,1)