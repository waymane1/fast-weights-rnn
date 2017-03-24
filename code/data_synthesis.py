import numpy as np
import string

def synthesize_key_value(n):
    '''
    Generates key value pairs to be used for training/testing.
    '''
    alpha = string.ascii_lowercase
    alpha_idx = np.random.choice(26, size=n, replace=False)
    alpha_array = [alpha[x] for x in alpha_idx]

    num_idx = np.random.randint(10, size=n)

    return (alpha_idx, num_idx)


def synthesize_sequence(size=3):
    '''
    Creates a string of alternating key value pairs.
    '''

    x, y = synthesize_key_value(size)
    sequence = np.insert(y, np.arange(len(x)), x)

    idx = np.random.choice(size, size=1, replace=False)

    return sequence


if __name__ == "__main__":
    print synthesize_sequence(5)
