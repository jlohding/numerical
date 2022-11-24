import numpy as np

class Integrator:
	def __init__(self, f: 'function'):
		self.f = f

	def compare(self, a:float, b:float, m:int = None) -> None:
		'''
		Runs all available numerical integration methods and prints results to console
		'''

		print(f"Trapezoid: {self.trapezoid(a,b)}")
		print(f"Simpson: {self.simpson(a,b)}")
		print(f"Midpoint: {self.midpoint(a,b)}")
		
		if m != None:
			print(f"Composite Trapezoid: {self.composite_trapezoid(a,b,m)}")
			print(f"Composite Simpson: {self.composite_midpoint(a,b,m)}")
			print(f"Composite Trapezoid: {self.composite_midpoint(a,b,m)}")
	
	def trapezoid(self, a:float, b:float) -> float:
		h = b-a
		return h/2 * (self.f(b) + self.f(a))

	def composite_trapezoid(self, a:float, b:float, m:int) -> float:
		h = (b-a) / m
		intervals = np.linspace(a,b,num=m,endpoint=False)[1:]

		return h/2 * (self.f(a) + self.f(b) + 2*sum(self.f(x) for x in intervals))

	def simpson(self, a:float, b:float) -> float:
		h = (b-a) / 2
		mid = a + h

		return h/3 * (self.f(a) + self.f(b) + 4*self.f(mid))

	def composite_simpson(self, a:float, b:float, m:int) -> float:
		h = (b-a)/(2*m)
		intervals = np.linspace(a,b,num=2*m,endpoint=False)[1:]
	
		f_vals = [self.f(x) for x in intervals]
		total = 0
		for i in range(len(f_vals)):
			if (i%2==0):
				total += f_vals[i] * 4
			else:
				total += f_vals[i] * 2
				
		return h/3 * (self.f(a) + self.f(b) + total) 

	def midpoint(self, a:float, b:float) -> float:
		h = b - a
		mid = a + h/2
		return h * self.f(mid)

	def composite_midpoint(self, a:float, b:float, m:int) -> float:
		h = (b-a) / m
		intervals = np.linspace(a,b, num=m, endpoint=False)
		intervals += h/2	
		
		return h * sum(self.f(x) for x in intervals)

	def adaptive_quadrature(self, a:float, b:float, tol:float = 0.01, method: 'string' = 'trapezoid') -> float:
		'''
		Recursively subdivide the interval and do trapezoid/simpson until error tolerance is met
		'''
		a_orig = a
		b_orig = b
		
		integration_methods = {
			"trapezoid" : self.trapezoid,
			"simpson" : self.simpson,
		}

		def quadrature(a,b, f: 'function'):
			m = (b-a)/2 + a

			S_ab = f(a,b)
			S_am = f(a,m) 
			S_mb = f(m,b) 

			if np.abs(S_ab - (S_am + S_mb)) < 3 * tol * (b-a)/(b_orig - a_orig):
				return (S_am + S_mb)
			else:
				return quadrature(a,m,f) + quadrature(m,b,f)
		
		return quadrature(a,b, integration_methods[method])

if __name__ == "__main__":
	integrator = Integrator(f = lambda x: 1/x)
	a, b, m = [1,3,4]

	integrator.compare(a,b,m)
	print("Adaptive Quadrature:", integrator.adaptive_quadrature(a,b, tol=0.001, method="simpson"))
