import numpy as np


def filter_degenerate_modes(frequencies, tolerance=1e-5):
    unique_indices = []
    skip_indices = set()

    for i, freq in enumerate(frequencies):
        if i in skip_indices or freq <= 0:  # Hoppa över akustiska/redan tagna
            continue

        unique_indices.append(i)

        # Leta efter en partner med nästan samma frekvens
        for j in range(i + 1, len(frequencies)):
            if abs(frequencies[j] - freq) < tolerance:
                skip_indices.add(j)
                break  # Vi hittade partnern, sluta leta för denna mod

    return unique_indices


if __name__ == "__main__":
    # Användning:
    phonons = np.load("/cfs/klemming/home/o/obollner/adaq/obollner/NV_64/data/phonon_data.npz")
    unique_mode_idxs = filter_degenerate_modes(phonons["freqs"])
    print(f"Hittade {len(unique_mode_idxs)} unika moder att simulera av totalt {len(phonons['freqs'])}.")
    print(unique_mode_idxs)
