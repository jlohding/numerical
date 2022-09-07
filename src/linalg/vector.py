from typing import Iterable

class Vector:
    def __init__(self, values: Iterable[float] or float, size: int = None):
        '''Constructor for 1-dimensional vector object

        Vectors are by default column vectors in this implementation. 
        To create a row vector, make a n x 1 matrix instead.

        values: Iterable[float] or int
            Iterable[float]:
                pre-populated vector of values, shape = 1 x len(values)
            float:
                scalar value for all entries of the vector

        size: int or None
            if type(values) == Iterable, size is not taken into consideration
            if values is a float, then the vector shape will be (size, 1)
        '''
        if isinstance(values, Iterable):
            self.vector = list(values)
            self.shape = (len(values), 1)
        elif isinstance(values, (float,int)):
            self.vector = [values for _ in range(size)]
            self.shape = (size, 1)
        else:
            raise Exception("Wrong input type")
    
    def get(self, i: int, j: int = None) -> float:
        '''Returns value of ith entry

        i: int
            row index

        j: int
            Ignored, not used, present for API consistency

        return: float
            value of ith entry of vector
        '''
        return self.to_list()[i]

    def set(self, i: int, val: float) -> 'Vector':
        '''Sets value at ith entry

        i: int
            ith index
        '''
        self.vector[i] = val
        return self
 
    def to_list(self):
        return self.vector

    def swap(self, i: int, j: int) -> 'Vector':
        '''Swaps elements
        
        i, j: int
            indices to swap
        '''
        self.vector[i], self.vector[j] = self.vector[j], self.vector[i]
        return self

    def dot(self, __o: 'Vector') -> float:
        '''Returns the dot product of this vector with another 

        __o: Vector
            Another vector class object
        
        return: float
            Returns the dot product of the two vectors
        '''
        if not isinstance(__o, Vector):
            raise Exception("Wrong input for dot product, must be Vector class")
        else:
            if len(self) != len(__o):
                raise Exception("Cannot calculate inner product for different lengths") 
            else:
                vals1 = self.to_list()
                vals2 = __o.to_list()
                return sum(vals1[i] * vals2[i] for i in range(len(self)))

    def p_norm(self, p: float = 2) -> float:
        '''Returns the vector p-norm
        
        p: float 
            Value of p for the computation
            For example: {'euclidean': 2, 'manhattan': 1, 'inf/sup' : float('inf')} 

        return: float
            Returns float value of p-norm
        '''

        if p == float('inf'):
            # sup norm, return largest abs value in vector
            return max(abs(x) for x in self.to_list())
        else:
            norm = sum(abs(x)**p for x in self.to_list()) ** (1/p)
            return norm

    def copy(self) -> 'Vector':
        # makes a copy of self
        return Vector([x for x in self.vector])

    def __len__(self) -> int:
        return self.shape[0]

    def __str__(self) -> str:
        '''Override __str__ to display vector
        '''
        return str(self.to_list())

    def __eq__(self, __o: object) -> bool:
        '''Override __eq__ to check elementwise equality
        '''
        if not isinstance(__o, Vector):
            return False
        else:
            return self.to_list() == __o.to_list()

    def __add__(self, __o: 'Vector') -> 'Vector':
        '''Override __add__ for vector elementwise addition
        '''
        if not isinstance(__o, Vector):
            raise Exception("Cannot add Vector with non-Vector")
        else:
            if len(self) != len(__o):
                raise Exception("Cannot add vectors of different length")
            else:
                return Vector([a + b for a,b in zip(self.to_list(), __o.to_list())])

    def __sub__(self, __o: 'Vector') -> 'Vector':
        '''Override __sub__ for vector elementwise subtraction
        '''
        if not isinstance(__o, Vector):
            raise Exception("Cannot subtract Vector with non-Vector")
        else:
            if len(self) != len(__o):
                raise Exception("Cannot subtract vectors of different length")
            else:
                return Vector([a - b for a,b in zip(self.to_list(), __o.to_list())]) 

    def __mul__(self, k: float) -> 'Vector':
        '''Override __mul__ for scalar multiplication
        '''
        if not isinstance(k, (int, float)):
            raise Exception("Must pass in scalar for scalar multiplication")
        else:
            return Vector([x*k for x in self.vector])

    def __rmul__(self, k: float) -> 'Vector':
        '''Override __rmul__ for scalar multiplication (commutative)
        '''
        return self.__mul__(k)