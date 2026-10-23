import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spinlabs.relaxation_dynamics import RelaxationDynamics

plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 14})

file_path = sys.argv[1:]
show_plot = False

ms_states = {"ms_1": [1.0, 0.0, 0.0], "ms_0": [0.0, 1.0, 0.0], "ms_-1": [0.0, 0.0, 1.0]}

init_state = "ms_0"

plt.figure(figsize=(10, 6))

if file_path[0] == "--plot":
    show_plot = True
    file_path = file_path[1:]

for file in file_path:
    print("processing file: ", file)
    simulator = RelaxationDynamics(init_state=ms_states[init_state], rates_data=file)
    T1_times = simulator.compute_T1_range()
    T1_room_temp = simulator.get_T1_fit(T=300)
    low_T = simulator.temperatures[1]
    T1_low_temp = simulator.get_T1_fit(T=low_T)
    print(10 * "=")
    print("T1 in room temp: ", T1_room_temp)
    print(f"Low temp (T={low_T}) limit: {T1_low_temp}")
    print(10 * "=")

    Path("T1_data").mkdir(exist_ok=True)
    filename = f"{simulator.defect}_T1_time_{init_state}"
    save_name = str(Path("T1_data") / filename)
    print("Saving T1 data in: ", save_name)
    np.savez(save_name)

    linestyle = "--" if "spin" in simulator.sim_type else "-"
    plt.plot(
        simulator.temperatures,
        T1_times,
        linewidth=2,
        linestyle=linestyle,
        label=f"{simulator.defect} {simulator.cell_size} {simulator.sim_type}",
    )

# plt.title(r"$T_1$ relaxation time")
plt.xlabel("Temperature (K)")
plt.ylabel(r"$T_1$ (s)")
plt.yscale("log")
plt.tick_params(axis="y", which="both", direction="in")

plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
if show_plot:
    plt.show()
else:
    save_name = f"figures/{simulator.defect}_T1_time_{init_state}"
    print("Saving figure in: ", save_name + ".png")
    plt.savefig(save_name)
