import numpy as np

def qrIteration(A, iters=50):
	'''
	Unshifted QR Iteration on matrix to converge to real Schur form
	'''
	if iters == 0:
		return A
	else:
		Q, R = np.linalg.qr(A)
		A2 = np.matmul(R,Q)
		return qrIteration(A2, iters-1)

def qrShiftedIteration(A, iters=50, shift_index=0):
	if iters == 0:
		return A
	else:
		shift = A.shape[0] - shift_index -1
		
		if shift < 0: # change this condition to ensure real schur form
			return A

		shift = A[shift][shift]
		A2 = A - (shift * np.eye(A.shape[0]))
		Q, R = np.linalg.qr(A2)
		A3 = np.matmul(R,Q) + (np.eye(A.shape[0]) * shift)
		
		return qrShiftedIteration(A3, iters-1, shift_index+1)

if __name__ == "__main__":
	A = np.array([[4,1,-2],[10,11,2],[0,2,-11]])

	print(np.linalg.eigvals(A))

	print(qrShiftedIteration(A).round(5))

	print(qrIteration(A, iters=3).round(5))

