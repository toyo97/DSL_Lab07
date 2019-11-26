import numpy as np
from spafe.features.gfcc import gfcc


def pad_to_dense(M):
    """Appends the minimal required amount of zeroes at the end of each
     array in the jagged array `M`, such that `M` looses its jagedness."""

    maxlen = max(r.size for r in M)

    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        Z[enu, :row.size] += row
    return Z


# use MinMaxScaler and make_pipeline
def transform(data, T=2**13, fs=8000, num_ceps=13):

    zero_padded_X = pad_to_dense(data)
    zero_padded_X /= np.max(np.abs(zero_padded_X), axis=1)[:, np.newaxis]

    zero_padded_X = zero_padded_X[:, :T]
    gfccs = gfcc(zero_padded_X[0], fs=fs, num_ceps=num_ceps)
    print(f'gfcc size: {gfccs.size}')
    X = np.zeros((zero_padded_X.shape[0], gfccs.size))
    print(f'Shape of X {X.shape}')

    for i in range(X.shape[0]):
        X[i, :] = gfcc(zero_padded_X[i, :], fs=fs, num_ceps=num_ceps).flatten()

    return X