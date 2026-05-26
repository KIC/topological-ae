import numpy as np
import sys


def calc_weights(number_of_weights=10, alpha=0.95):
    """
    Return decaying weights [w0, w1, ..., w_{n-1}] such that:
        - w0 = 1.0 (strongest, for shortest window)
        - weights decay geometrically by factor `alpha` per step
        - all weights are normalized to sum to 1.0
    """
    # geometric decay: [1, alpha, alpha**2, ...]
    exponents = np.arange(number_of_weights)
    raw_weights = alpha ** exponents        # 1.0, alpha, alpha^2, ...

    # normalize so they sum to 1
    weights = raw_weights # / raw_weights.sum()
    return weights


if __name__ == "__main__":
    print(f"Usage: {sys.argv[0]} <soure.npy> <alpha default 0.95>")

    # Load data
    file_name = sys.argv[1]
    print("Load array", file_name)
    arr = np.load(file_name)

    # compute weights
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95

    weights = calc_weights(arr.shape[1], alpha=alpha).astype("float32")

    print("apply weights", weights.shape, arr.shape)
    arr *= weights

    np.save(f"{file_name}.weighted.npy", arr)

    print("done")
