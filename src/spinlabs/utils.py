import numpy as np
from scipy import constants as Cn
from typing import Tuple


class MathUtils:
    @staticmethod
    def broad_delta(omega, omega_p, sigma):
        delta = np.exp(-0.5 * (omega_p - omega) ** 2 / (sigma**2)) * 1 / (np.sqrt(2 * Cn.pi) * sigma) ** omega.ndim
        s = delta.sum()
        if s > 0:
            delta /= s
        return delta

    @staticmethod
    def gaussian_density(x, x0, width):
        """Normalised Gaussian in ``x`` around ``x0``: a broadened delta(x - x0) in units of 1/[x].

        Unlike :meth:`broad_delta` it is not rescaled to sum to 1 on the grid, so
        ``sum(f * gaussian_density(...)) * dx`` approximates the integral of f
        against the delta independently of the grid spacing.
        """
        return np.exp(-0.5 * ((x - x0) / width) ** 2) / (np.sqrt(2 * Cn.pi) * width)

    @staticmethod
    def expand_data(freqs: np.ndarray, values: np.ndarray, res: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        freqs = np.array(freqs)
        values = np.array(values)
        f_min = 0
        f_max = np.max(freqs) + 5 * sigma
        x_grid = np.arange(f_min, f_max, res)

        y_dense = np.zeros_like(x_grid)
        indices = np.round(freqs / res).astype(int)
        for i, idx in enumerate(indices):
            if 0 <= idx < len(y_dense):
                y_dense[idx] += values[i]

        return x_grid, y_dense

    @staticmethod
    def smear_data(freqs: np.ndarray, values: np.ndarray, res: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        x_grid, y_dense = MathUtils.expand_data(freqs, values, res, sigma)

        x_kernel = np.arange(-4 * sigma, 4 * sigma + res, res)
        kernel = np.exp(-0.5 * (x_kernel / sigma) ** 2) / (sigma * np.sqrt(2 * Cn.pi))

        y_smooth = np.convolve(y_dense, kernel, mode="same")
        if len(y_smooth) > len(x_grid):
            diff = len(y_smooth) - len(x_grid)
            start = diff // 2
            y_smooth = y_smooth[start : start + len(x_grid)]

        return x_grid, y_smooth

    @staticmethod
    def expand_data_2d(freqs, values, res, sigma):
        # TODO: debugging
        # print("Expanding data...")
        freqs = np.array(freqs)

        values = np.array(values)

        values = values.flatten()

        f_min = 0
        f_max = np.max(freqs) + 5 * sigma

        x_array = np.arange(f_min, f_max, res)
        y_array = np.arange(f_min, f_max, res)
        # print("Grid size:")
        # print(x_array.shape, y_array.shape)
        dense_values = np.zeros((len(x_array), len(y_array)))

        freq_x, freq_y = np.meshgrid(freqs, freqs)
        freq_x_flat = freq_x.flatten()
        freq_y_flat = freq_y.flatten()

        idx_x = np.round(freq_x_flat / res).astype(int)
        idx_y = np.round(freq_y_flat / res).astype(int)

        mask = (idx_x >= 0) & (idx_x < len(x_array)) & (idx_y >= 0) & (idx_y < len(y_array))

        np.add.at(dense_values, (idx_x[mask], idx_y[mask]), values[mask])

        return x_array, y_array, dense_values

    @staticmethod
    def get_2d_spectral_density(freqs, values, res, sigma_phys):
        """Pair weights ``values[i, j]`` at (freqs[i], freqs[j]), smeared by ``sigma_phys`` on each axis.

        Returns the grid and the spectral density per unit frequency squared
        (like :meth:`smear_data` in 1D), so its integral does not depend on ``res``.
        """
        from scipy.ndimage import gaussian_filter

        x_array, y_array, dense_values = MathUtils.expand_data_2d(freqs, values, sigma=sigma_phys, res=res)

        sigma_pixel = sigma_phys / res
        # Use separable 2D gaussian_filter instead of 2D convolve with a 121x121 kernel
        # (O(N) vs O(N^2), 100x faster, identical results down to 1e-17, prevents OOM)
        smoothed = gaussian_filter(dense_values, sigma=sigma_pixel, mode="constant", cval=0.0)

        X, Y = np.meshgrid(x_array, y_array)

        # The filter keeps each pixel's summed weight; per pixel area that is a density.
        return X, Y, smoothed / res**2

    @staticmethod
    def rotation_matrix_around_axis(axis, angle):
        """
        Compute the 3x3 rotation matrix for a rotation by `angle` (in radians)
        around the given axis.

        Parameters
        ----------
        axis : array_like (3,)
            The rotation axis. It will be normalized internally.
        angle : float
            Rotation angle in radians (right-hand rule).

        Returns
        -------
        R : numpy.ndarray (3,3)
            Rotation matrix.
        """
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        omc = 1 - cos_a  # 1 - cos(angle)

        R = np.array(
            [
                [cos_a + x * x * omc, x * y * omc - z * sin_a, x * z * omc + y * sin_a],
                [y * x * omc + z * sin_a, cos_a + y * y * omc, y * z * omc - x * sin_a],
                [z * x * omc - y * sin_a, z * y * omc + x * sin_a, cos_a + z * z * omc],
            ]
        )
        return R

    @staticmethod
    def rotation_around_symmetry_axis(eigvec, order=3):
        """
        Return the rotation matrix for a C<order> rotation around the given
        eigenvector (symmetry axis).

        eigvec : array_like (3,)
            The “z‑axis” eigenvector (e.g., the one with unique eigenvalue).
        order  : int
            Rotation order (2 → 180°, 3 → 120°, etc.).
        """
        angle = 2 * np.pi / order
        return MathUtils.rotation_matrix_around_axis(eigvec, angle)

    @staticmethod
    def reflection_matrix(plane_normal):
        """
        Return the 3x3 Householder matrix that reflects a vector
        across the plane whose unit normal is `plane_normal`.

        Parameters
        ----------
        plane_normal : array_like (3,)
            A vector perpendicular to the mirror plane. It will be normalised.

        Returns
        -------
        M : numpy.ndarray (3,3)
        """
        n = np.asarray(plane_normal, dtype=float)
        n = n / np.linalg.norm(n)
        return np.eye(3) - 2.0 * np.outer(n, n)

    @staticmethod
    def calc_ipr(eigenvectors: np.ndarray) -> np.ndarray:
        """
        Calculate the Inverse Participation Ratio (IPR) for each phonon mode.

        The IPR measures how localized a mode is over the atoms of the supercell:

            IPR = sum_j ( |e_j|^2 )^2 / ( sum_j |e_j|^2 )^2

        where |e_j|^2 is the squared displacement amplitude (summed over x,y,z)
        for atom j. For a mode perfectly localized on a single atom IPR = 1,
        while for a mode uniformly delocalized over N atoms IPR = 1/N.

        Parameters
        ----------
        eigenvectors : array_like, shape (n_modes, n_atoms, 3)
            Mass-weighted phonon eigenvectors.

        Returns
        -------
        numpy.ndarray, shape (n_modes,)
            The IPR value for each mode.
        """
        eigs = np.asarray(eigenvectors, dtype=float)
        # |e_j|^2 per atom (sum over cartesian components), shape (n_modes, n_atoms)
        disp_sq = np.sum(eigs**2, axis=-1)

        disp_total = np.sum(disp_sq, axis=-1)
        disp_4 = np.sum(disp_sq**2, axis=-1)

        return disp_4 / disp_total**2

    @staticmethod
    def calc_locality_weight(
        eigenvectors: np.ndarray,
        frac_coords: np.ndarray,
        lattice: np.ndarray,
        defect_pos: np.ndarray = (0.0, 0.0, 0.0),
        radius: float = 2.0,
    ) -> np.ndarray:
        """
        Calculate the fraction of each mode's displacement that lies within
        `radius` (in Angstrom) of the defect position (locality weight).

        A value close to 1 means the mode is localized around the defect,
        while a value close to 0 means the mode lives in the bulk.

        Parameters
        ----------
        eigenvectors : array_like, shape (n_modes, n_atoms, 3)
            Mass-weighted phonon eigenvectors.
        frac_coords : array_like, shape (n_atoms, 3)
            Fractional atomic coordinates (already shifted so the defect
            is at the origin, e.g. from PhononManager.translate_defect_to_origin).
        lattice : array_like, shape (3, 3)
            Lattice vectors as rows.
        defect_pos : array_like, shape (3,), optional
            Fractional coordinate of the defect centre. Default is the origin.
        radius : float, optional
            Cutoff radius in Angstrom around the defect. Default is 2.0.

        Returns
        -------
        numpy.ndarray, shape (n_modes,)
            Locality weight in [0, 1] for each mode.
        """
        eigs = np.asarray(eigenvectors, dtype=float)
        frac_coords = np.asarray(frac_coords, dtype=float)
        lattice = np.asarray(lattice, dtype=float)
        defect_pos = np.asarray(defect_pos, dtype=float)

        # Minimum-image displacement vectors from the defect, in fractional coords
        diff_frac = np.mod(frac_coords - defect_pos + 0.5, 1.0) - 0.5
        cart_dist = np.linalg.norm(diff_frac @ lattice, axis=-1)

        within = cart_dist < radius

        disp_sq = np.sum(eigs**2, axis=-1)  # (n_modes, n_atoms)
        disp_total = np.sum(disp_sq, axis=-1)  # (n_modes,)
        disp_local = np.sum(disp_sq[:, within], axis=-1)

        return disp_local / disp_total

    @staticmethod
    def fmt(val, precision=6):
        """Return a nicely formatted string for either a number or a tuple of numbers."""
        if isinstance(val, tuple):
            # For tuples, show each element with the given precision,
            # separated by commas, inside parentheses.
            return "(" + ", ".join(f"{v:.{precision}f}" for v in val) + ")"
        else:
            # Assume it's a single number (float/int).
            return f"{val:.{precision}f}"

    @staticmethod
    def calc_symmetry(sym_a: str, sym_b: str) -> list[str]:
        """Calculate resulting symmetry irreps for the C3v point group."""

        def parse_irrep(sym: str) -> str:
            sym_upper = sym.upper()
            if sym_upper in ("EX", "EY"):
                return "E"
            return sym_upper

        a_parsed = parse_irrep(sym_a)
        b_parsed = parse_irrep(sym_b)

        product_table = {
            ("A1", "A1"): ["A1"],
            ("A1", "A2"): ["A2"],
            ("A1", "E"): ["E"],
            ("A2", "A1"): ["A2"],
            ("A2", "A2"): ["A1"],
            ("A2", "E"): ["E"],
            ("E", "A1"): ["E"],
            ("E", "A2"): ["E"],
            ("E", "E"): ["A1", "A2", "E"],
        }

        pair = (a_parsed, b_parsed)
        if pair in product_table:
            return product_table[pair]

        reverse_pair = (b_parsed, a_parsed)
        if reverse_pair in product_table:
            return product_table[reverse_pair]

        raise ValueError(f"Direct product for irreps {sym_a} and {sym_b} is not defined.")
