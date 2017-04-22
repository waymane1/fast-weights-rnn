import copy, numpy as np

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_derivative(output):
    return output*(1-output)

def update_layers():
    for bit in range(digits):

        X = np.array([[a[digits - bit - 1],b[digits - bit - 1]]])
        y = np.array([[c[digits - bit - 1]]]).T

        layer_1 = sigmoid(np.dot(X,W_0) + np.dot(layer_1[-1],W_h))

        layer_2 = sigmoid(np.dot(layer_1,W_1))

        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_derivative(layer_2))
        total_error += np.abs(layer_2_error[0])

        d[digits - bit - 1] = np.round(layer_2[0][0])

        layer_1.append(copy.deepcopy(layer_1))

    layer_1_delta_next = np.zeros(hidden_dim)

def update_weights():
    for bit in range(digits):

        X = np.array([[a[bit],b[bit]]])

        layer_1 = layer_1[-bit-1]
        layer_1_prev = layer_1[-bit-2]

        # error: output layer
        layer_2_delta = layer_2_deltas[-bit-1]

        # error: hidden layer
        layer_1_delta = (layer_1_delta_next.dot(W_h.T) + layer_2_delta.dot(W_1.T)) * sigmoid_derivative(layer_1)

        W_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        W_h_update += np.atleast_2d(layer_1_prev).T.dot(layer_1_delta)
        W_0_update += X.T.dot(layer_1_delta)

        layer_1_delta_next = layer_1_delta


np.random.seed(0)
num_iterations = 1000
binary_map = {}
digits = 8

largest_number = pow(2,digits)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    binary_map[i] = binary[i]

# "Learning rate"
alpha = 0.1

# dimensions
input_dim = 2
hidden_dim = 16
output_dim = 1


# initialize weights
W_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
W_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
W_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

# these are updates to make at each iteration to the above
W_0_update = np.zeros_like(W_0)
W_1_update = np.zeros_like(W_1)
W_h_update = np.zeros_like(W_h)

for j in range(num_iterations):

    a_int = np.random.randint(largest_number/2)
    a = binary_map[a_int]

    b_int = np.random.randint(largest_number/2)
    b = binary_map[b_int]

    c_int = a_int + b_int
    c = binary_map[c_int]

    d = np.zeros_like(c)

    total_error = 0

    layer_2_deltas = list()
    layer_1 = list()
    layer_1.append(np.zeros(hidden_dim))

    update_layers()

    update_weights()

    W_0 += W_0_update * alpha
    W_1 += W_1_update * alpha
    W_h += W_h_update * alpha

    W_0_update *= 0
    W_1_update *= 0
    W_h_update *= 0

    if(j % 1000 == 0):
        print "###" + str(j) + "###"
        print "Error:" + str(total_error)
        print "Network:" + str(d)
        print "Answer:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x * pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
