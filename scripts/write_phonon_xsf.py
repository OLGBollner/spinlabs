#!/bin/python
import argparse

import numpy as np

parser = argparse.ArgumentParser("Writes the data file for a specific phonon mode")
parser.add_argument("phonon_data", metavar="phonon_data", help="Phonon data file")
parser.add_argument("modes", metavar="modes", nargs="+", type=int, help="Modes to write to file (indexing from 1)")
parser.add_argument("-scale", dest="scale", type=float, default=1, help="Scaling of eigenvectors (1 default)")
args = parser.parse_args()

data = np.load(args.phonon_data)
lattice = data["lattice"]
species = data["atom_symbols"]
atoms = data["atoms"]
eigs = data["eigs"]

weights = {"C": 12.011, "Si": 28.0855, "Cl": 35.453, "B": 10.811, "N": 14.0067}
mass_dist = np.zeros(3)
# normal_mass_dist = np.zeros(3)

# for i in range(len(atoms)):
#    pos = np.matmul(np.transpose(lattice), np.transpose(atoms[n,:]))
#    normal_mass_dist

for mode in args.modes:
    mode_eigs = eigs[mode - 1, :, :]
    file_name = f"phonon_{mode}.xsf"
    content = []
    content.append("CRYSTAL\n")
    content.append("PRIMVEC\n")
    for n in range(lattice.shape[0]):
        content.append("{} {} {}\n".format(*lattice[n, :]))
    content.append("PRIMCOORD\n")
    content.append(f"{len(atoms)} 1\n")
    mass_dist = np.zeros(3)
    for n in range(len(atoms)):
        mass = weights[species[n]]
        if n != 574:
            mass_dist += mode_eigs[n, :] * np.sqrt(mass)
        # if not (species[n] == "C" or species[n] == "H"):
        #   continue
        coord_force = list(np.matmul(np.transpose(lattice), np.transpose(atoms[n, :]))) + list(
            np.transpose(mode_eigs[n, :] / np.sqrt(mass))
        )
        content.append("{} {} {} {} {} {} {}\n".format(species[n], *coord_force))
    print("Mode {}: p = {}, {}, {}".format(mode, *list(mass_dist)))

    with open(file_name, "w") as writer:
        writer.writelines(content)
