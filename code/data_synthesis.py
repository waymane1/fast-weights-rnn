import numpy as np
import string

def synthesize_key_value(n):
    '''
    Generates n keys and n values to be used for training/testing.
    '''
    alpha = string.ascii_lowercase
    alpha_idx = np.random.choice(26, size=n, replace=False)
    alpha_array = [alpha[x] for x in alpha_idx]

    num_idx = np.random.random_integers(26, high=35, size=n)

    return (alpha_idx, num_idx)


def synthesize_sequence(size=3):
    '''
    Creates an array of alternating key value pairs.
    '''

    x, y = synthesize_key_value(size)
    sequence = np.insert(y, np.arange(len(x)), x)

    # Generate the random index we will use to query
    # choice() returns an array. Since we only need one, return 0th index.
    idx = np.random.choice(size, size=1, replace=False)[0]
    sol = y[idx]

    query_list = [36, 36, x[idx]]
    sequence = np.append(sequence, np.array(query_list))

    return sequence, sol


def one_hot(seq, size=3, char_set_size=37):
    '''
    Converts a given sequence into it's one-hot encoding
    '''
    num_columns = 37
    num_rows = (2*size) + 3

    ohot_matrix = np.zeros((num_rows, num_columns))
    ohot_matrix[np.arange(num_rows), seq] = 1

    return ohot_matrix


def debug_display_sequence(seq, characters = None):
    '''
    Outputs the given numerical sequence as a string.
    '''
    output = ""

    for i, x in enumerate(seq):
        output += characters[x]

    return output

if __name__ == "__main__":
    sequence, sol = synthesize_sequence(5)
    char_set = list(string.lowercase[:]) + [str(x) for x in range(10)] + ["?"]
    char_sequence = debug_display_sequence(sequence, char_set)

    print sequence, char_sequence, sol
    print one_hot(sequence, size=5, char_set_size=len(char_set))
