from argparse import ArgumentParser as Parser
import numpy as np
import matplotlib.pyplot as plt
from utils import smear_data

if __name__ == "__main__":
    parser = Parser(description="Plots ZFS phonon derivatives.")
    parser.add_argument("zfs_data", metavar="zfs_data", nargs="+", help="npz files containing zfs derivatives")
    parser.add_argument("-o", "--output", help="Specify the output filename for the plot or data")
    parser.add_argument("-i", "--ipr", action="store_true", help="Specify the IPR lower limit for the plot")
    parser.add_argument("-f", "--format", help="File format of plot")
    parser.add_argument("-d", "--difference", action="store_true", help="Calculate difference between data")
    parser.add_argument("-n", "--norm", action="store_true", help="plot norm")
    parser.add_argument("-b", "--bar", action="store_true", help="bargraph of mode indices")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the figure directly without saving.")

    args = parser.parse_args()

    zfs_files = args.zfs_data

    colors = iter(["red", "black", "blue", "orange", "green"])

    plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10})

    fig, ax = plt.subplots(figsize=(6, 5))
    ax2 = ax.twinx()

    zfs_data = [np.load(file) for file in zfs_files]

    plot_name = ""
    if args.difference:
        data = {
            key: np.abs(zfs_data[1][key] - zfs_data[0][key])
            for key in zfs_data[0].keys()
            if key != "ipr" and key != "freqs"
        }
        data["ipr"] = zfs_data[0]["ipr"]
        data["freqs"] = zfs_data[0]["freqs"]
        zfs_data = [data]
        plot_name = r"$\Delta$"

    for i, data in enumerate(zfs_data):
        pert_scale = zfs_files[i].split("_")[-2]

        filename = ""
        if args.output:
            filename = args.output
        else:
            if args.norm:
                filename += "zfs_norm_vs_pert"
            else:
                filename += "zfs_vs_pert"

            filename += "_{}".format(pert_scale)

            if args.bar:
                filename += "_bar"

            if args.ipr:
                filename += "_ipr"

            if args.difference:
                filename += ":diff"
            else:
                sim_type = "all_bands" if "all" in zfs_files[i] else "defect_band_approx"

                filename += ":{}".format(sim_type)
        filename += args.format if args.format else ".png"

        ipr = data["ipr"]
        freqs = data["freqs"]
        zfs_derivs = data["zfs_derivs"]
        V_0_0 = data["V_0_0"]
        V_0_pm = data["V_0_pm"]
        V_p_m = data["V_p_m"]

        sigma = 7.5
        if args.norm:
            color = next(colors)
            coupling_strength = np.linalg.norm(zfs_derivs, axis=(1, 2))

            if args.bar:
                ax2.bar(
                    np.array(range(len(coupling_strength))),
                    coupling_strength,
                    color=color,
                    alpha=0.6,
                    label=plot_name + r"$|\partial D^{(1)}|$",
                )
            else:
                if not args.ipr:
                    ax.vlines(
                        freqs, [0], coupling_strength, color=color, alpha=0.6, label=plot_name + r"$|\partial D^{(1)}|$"
                    )

                smooth_x, smooth_y = smear_data(freqs, coupling_strength, 1, sigma)
                ax2.plot(
                    smooth_x, smooth_y, color=color, linewidth=2, label=plot_name + r"$F^{(1)}$" + f" {sim_type[0]}"
                )
        else:
            if not args.ipr:
                ax.vlines(freqs, [0], V_0_pm / np.sqrt(2), label=plot_name + r"$V_{0\pm}^l$", color="blue", alpha=0.6)
                ax.vlines(freqs, [0], V_p_m, label=plot_name + r"$V_{+-}^l$", color="red", alpha=0.6)
                ax.vlines(freqs, [0], V_0_0 / 3, label=plot_name + r"$V_{00}^l$", color="black", alpha=0.6)

            smooth_x, smooth_y = smear_data(freqs, V_0_pm / np.sqrt(2), 0.5, sigma)
            ax2.plot(smooth_x, smooth_y, color="blue", linewidth=2, label=plot_name + r"$F_{0\pm}^{(1)}$")

            smooth_x, smooth_y = smear_data(freqs, V_p_m, 0.5, sigma)
            ax2.plot(smooth_x, smooth_y, color="red", linewidth=2, label=plot_name + r"$F_{+-}^{(1)}$")

            smooth_x, smooth_y = smear_data(freqs, V_0_0 / 3, 0.5, sigma)
            ax2.plot(smooth_x, smooth_y, color="black", linewidth=2, label=plot_name + r"$F_{00}^{(1)}$")

        if args.ipr:
            if args.bar:
                ax.bar(np.array(range(len(ipr))), ipr, color="blue", alpha=0.6, label="IPR")
            else:
                smooth_x, smooth_y = smear_data(freqs, ipr, 1, sigma)
                ax.plot(smooth_x, smooth_y, color="blue", alpha=0.6, label="IPR")
            ax.set_ylabel("Phonon IPR")
        else:
            ax.set_ylabel("Coupling coefficient (MHz)")
            ax.set_ylim(0, 200)

        if args.bar:
            ax.set_xlabel("Mode index")
            ax2.set_ylabel("Coupling coefficient (MHz)")
            legend_pos = "upper right"
        else:
            ax.set_xlim(0, 200)
            ax.set_xlabel("Vibration frequency (meV)")
            legend_pos = "upper left"
            ax2.set_ylabel(r"Spectral function (MHz$^2$)")
            ax2.set_ylim(0, 2000)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax.legend(lines1 + lines2, labels1 + labels2, loc=legend_pos, frameon=True)

        plt.tight_layout()

        if args.plot:
            plt.show()
        else:
            print("Saving figure in: ", "figures/" + filename)
            plt.savefig("figures/" + filename)
