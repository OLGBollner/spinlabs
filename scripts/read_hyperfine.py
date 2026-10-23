import argparse

import numpy as np

parser = argparse.ArgumentParser("Reads hyperfine parameters and vectors of desired ions")
parser.add_argument("hyp_file", metavar="hyp_file", help="Hyperfine data file")
parser.add_argument("ions", metavar="ions", type=int, nargs="+", help="Which ions to read (start from 1)")
parser.add_argument(
    "--xyz", dest="xyz", default=False, action="store_true", help="Print the hyperfine matrix in xyz basis"
)
parser.add_argument(
    "--tensor",
    dest="tensor",
    default=False,
    action="store_true",
    help="Print the hyperfine matrix, as matrix, in xyz basis",
)
parser.add_argument(
    "--pos", dest="pos", default=False, action="store_true", help="Print the position in cartesian coordinates"
)
args = parser.parse_args()

# data in format (ions, (0 if diagonalized matrix, 1 if xyz matrix, 2 if eigenvectors columnwise), 3x3-matrix)
data = np.load(args.hyp_file, allow_pickle=True)


for ion in args.ions:
    print(f"For ion {ion}:")
    if args.xyz:
        mat = data[ion - 1, 1, :, :]
        print("Tensor:")
        print(f"{mat[0, 0]} {mat[1, 1]} {mat[2, 2]} {mat[0, 1]} {mat[0, 2]} {mat[1, 2]}")
    if args.tensor:
        print("Tensor:")
        mat = data[args.ions[0] - 1, 1, :, :]
        for i in range(3):
            for j in range(3):
                print(f"{mat[i, j]:<4.3f}", end=" ")
            print()
    if args.pos:
        print(f"Position: {np.diag(data[ion - 1, 3, :, :])}")
    eigs = np.diag(data[ion - 1, 0, :, :])
    for label, eig in zip(["A1", "A2", "A3"], eigs):
        print(f"{label}: {eig:.3g} MHz")
    print("Vectors:")
    print(data[ion - 1, 2, :, :])
    print()
