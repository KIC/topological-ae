import numpy as np
import sys


if __name__ == "__main__":
    print(f"Usage: {sys.argv[0]} <soure.npy> <sub_sample_size default None> <extreme_event_threshold default 2.5> <putfile.npy deafault post.npy>")

    # Load data
    file_name = sys.argv[1]
    print("Load array", file_name)
    arr = np.load(file_name)

    # sub samplig parameter
    sub_sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    extreme_event_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5

    # sub sample if requested
    if sub_sample_size:
        print("sub sampling", sub_sample_size)
        max_z_per_row = np.max(np.abs(arr), axis=1)

        threshold = extreme_event_threshold
        extreme_mask = max_z_per_row > threshold

        # 3. Indizes für beide Töpfe extrahieren
        extreme_indices = np.where(extreme_mask)[0]
        normal_indices = np.where(~extreme_mask)[0]
        print("extreme events:", len(extreme_indices))

        sampled_normal_indices = np.random.choice(len(normal_indices), size=sub_sample_size, replace=False)

        final_train_indices = np.concatenate([extreme_indices, sampled_normal_indices])
        np.random.shuffle(final_train_indices) # Mischen, damit das Netz nicht erst nur Anomalien sieht

        subset = arr[final_train_indices]
        print("keept exptrem and normal samples", len(subset))
    else:
        np.random.shuffle(arr)
        subset = arr

    print("save subsampled and shuffled array", subset.shape)
    np.save(sys.argv[4] if len(sys.argv) > 4 else f"{file_name}.shuffled.npy", subset)

    print("done")
