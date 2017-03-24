import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def nonlin_sigmoid(x, deriv=False):
	if (deriv == True):
		return x * (1-x)
	return 1/(1 + np.exp(-x))

def const_data_gen():
	X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
	y = np.array([[0,0,1,1]]).T
	return (X, y)

if __name__ == "__main__":
	x = np.linspace(-5, 4, 10, endpoint=True)
	y = np.array([])
	for i in range(-5, 5):
		value = nonlin_sigmoid(i)
		y = np.append(y, value)
		print value
	plt.rcParams['toolbar'] = 'None'
	plt.plot(x, y)
	plt.show()

	x,y = const_data_gen()

	np.random.seed(1)

	w0 = 2 * np.random.random((3,1)) - 1

	for i in xrange(1000):
		lay0 = x
		lay1 = nonlin_sigmoid(np.dot(lay0, w0))

		lay1_error = y - lay1

		lay1_delta = lay1_error * nonlin_sigmoid(lay1, True)

		w0 += np.dot(lay0.T, lay1_delta)

	print "O/P after training"
	print lay1
