import sys

import matplotlib.pyplot as plt
import numpy as np

from spinlabs.relaxation_dynamics import RelaxationDynamics

file_path = sys.argv[1]

ms_states = {"ms_1": [1.0, 0.0, 0.0], "ms_0": [0.0, 1.0, 0.0], "ms_-1": [0.0, 0.0, 1.0]}

init_state = "ms_-1"

plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 14})


plt.figure(figsize=(8, 6))

for target_temperature in [300]:  # np.linspace(1e-2, 20, 3):
    simulator = RelaxationDynamics(init_state=ms_states[init_state], rates_data=file_path)

    T1_eigenval, eigenvalues, eigenvectors = simulator.get_T1_eigenval(T=target_temperature)
    steady_state = simulator.get_steady_state(T=target_temperature)[simulator.population.argmax()]

    time_points = np.linspace(0, 10 * T1_eigenval + 1 / target_temperature, 10000)

    population = simulator.simulate_relaxation(time_points, T=target_temperature)

    polarized_pop = population[:, simulator.population.argmax()]
    T1_fit = simulator.fit_T1_exp(polarized_pop, time_points, steady_state, T1_eigenval)

    # print("T1 population: ", T1_fit)
    # print("T1 eigenvalue: ", T1_eigenval)

    plt.plot(time_points, population[:, 0], label=r"$m_s=1$", linewidth=3)
    plt.plot(time_points, population[:, 1], label=r"$m_s=0$", linewidth=3)
    plt.plot(time_points, population[:, 2], label=r"$m_s=-1$", linewidth=3)

    n_markers = 10
    log_indices = np.unique(np.logspace(0, np.log10(len(time_points) - 1), n_markers).astype(int))
    plt.plot(
        time_points,
        polarized_pop,
        label="Exp Fit",
        linewidth=2,
        linestyle="--",
        marker="x",
        markersize=10,
        markevery=list(log_indices),
    )

    plt.plot(time_points, np.sum(population, axis=1), label="Total Population", linestyle="--", color="black")
    plt.axvline(x=T1_fit, label=r"$T_1=$" + f"{T1_fit:.3f} s", linestyle="--", color="gray")

plt.yscale("log")
plt.xscale("log")

# plt.title(f"Population Relaxation Dynamics at T = {target_temperature} K")
plt.xlabel("Time (s)")
plt.ylabel("Population")
plt.grid(True, linestyle=":", alpha=0.6)
plt.xlim(time_points[0], time_points[-1])
plt.legend(loc="best")
plt.tight_layout()
save_name = f"figures/{simulator.defect}_{simulator.cell_size}_population_state_{init_state}_{target_temperature}"
plt.savefig(save_name)
plt.show()
