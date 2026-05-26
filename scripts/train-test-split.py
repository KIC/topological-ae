from sklearn.model_selection import train_test_split
import numpy as np
import sys


if __name__ == "__main__":
    print(f"Usage: {sys.argv[0]} <soure.npy> <test_size default 0.2>")

    # Load data
    file_name = sys.argv[1]
    print("Load array", file_name)
    arr = np.load(file_name)

    test_size = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    train, test = train_test_split(arr, test_size=test_size)

    print("save split arrays", train.shape, test.shape)
    np.save(f"{file_name}.train.npy", train)
    np.save(f"{file_name}.test.npy", test)

    print("done")
