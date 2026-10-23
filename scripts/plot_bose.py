import sys

import numpy as np
from matplotlib import pyplot as plt

from spinlabs.transition_rate import Phonons

data = np.load(sys.argv[1], allow_pickle=True)
omega = data["freqs"]

T = 100

n = Phonons.bose_einstein(omega, T)

omega_x, omega_y = np.meshgrid(omega, omega)
print("omega_x: ", omega_x.shape)

n2 = Phonons.bose_einstein_2d(omega_x[0, :], T)
print("n2: ", n2["abs_em"].shape)

fig, ax = plt.subplots()
ax.plot(omega, n)
fig2, ax2 = plt.subplots()

ax2.set_xlabel(r"$\omega_l$")
ax2.set_ylabel(r"$\omega_{l'}$")
mesh = ax2.pcolormesh(omega_x, omega_y, n2["abs_abs"], cmap="viridis", shading="auto")
cbar = fig2.colorbar(mesh, ax=ax2)

plt.show()
