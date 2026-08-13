# Python file to plot important quanities
import matplotlib.pyplot as plt
import numpy as np
import os


#######################################
##### BATCH LEVEL DATA PLOTTING #######
#######################################
def Shannon_Entropy(batches, entropy, save_path):
    entropy = np.array(entropy)
    batches = np.array(batches)

    # Compute running average
    running_avg = np.cumsum(entropy) / np.arange(1, len(entropy) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(batches, entropy, marker="o", color="blue", label="Entropy")
    plt.plot(
        batches,
        running_avg,
        marker="x",
        color="red",
        linestyle="--",
        label="Running Average",
    )
    plt.xlabel("Batch Number")
    plt.ylabel("Shannon Entropy")
    plt.title("Source Convergence (Shannon Entropy)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def K_Effective(batches, keff, save_path="keff_plot.png"):
    batches = np.array(batches)
    keff = np.array(keff)

    # Compute running average
    running_avg = np.cumsum(keff) / np.arange(1, len(keff) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(batches, keff, marker="o", color="blue", label="k-effective")

    plt.plot(
        batches,
        running_avg,
        marker="x",
        color="red",
        linestyle="--",
        label="Running Average",
    )
    plt.xlabel("Batch Number")
    plt.ylabel(r"$k_{eff}$")
    plt.title("k-effective Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


#######################################
###### STEP LEVEL DATA PLOTTING #######
#######################################
def plot_generated_data(nuclides, data, save_folder, worker_id):
    save_folder += f"/plots/worker{worker_id}"
    os.makedirs(save_folder, exist_ok=True)
    time = data["time_days"]
    # Categorize nuclides
    actinides = [
        nuc for nuc in nuclides if nuc.startswith(("U", "Pu", "Np", "Am", "Cm"))
    ]
    fission_products = [nuc for nuc in nuclides if nuc not in actinides]

    # Plot actinides
    if actinides:
        plt.figure(figsize=(12, 6))
        for nuclide in actinides:
            concentration = data[nuclide]
            plt.plot(time, concentration, marker="o", linewidth=2, label=nuclide)

        plt.xlabel("Time [d]")
        plt.ylabel("Number density [atom/b-cm]")
        plt.title("Actinide Number Density Evolution")
        plt.yscale("log")
        plt.grid(True, alpha=0.3, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "actinides_number_density.png"), dpi=300)

    # Plot fission products, these are all 0 at t=0
    if fission_products:
        plt.figure(figsize=(12, 6))
        for nuclide in fission_products:
            concentration = data[nuclide]
            plt.plot(
                time[1:], concentration[1:], marker="o", linewidth=2, label=nuclide
            )

        plt.xlabel("Time [d]")
        plt.ylabel("Number density [atom/b-cm]")
        plt.title("Fission Product Number Density Evolution")
        plt.yscale("log")
        plt.grid(True, alpha=0.3, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_folder, "fission_products_number_density.png"), dpi=300
        )

    # ---------- Additional Reactor Parameters ----------
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        time,
        data["k_eff"],
        yerr=data["k_eff_std"],
        marker="o",
        linewidth=2,
        label="k_eff",
    )
    plt.xlabel("Time [d]")
    plt.ylabel("Effective Multiplication Factor (k_eff)")
    plt.title("keff vs Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "k_eff.png"), dpi=300)
    plt.close()

    parameters = {
        "power_W_g": "Reactor Power [W/g]",
        "int_p_W": "Integrated Power [W/ g]",
        "fuel_temp_K": "Fuel Temperature [K]",
        "mod_temp_K": "Moderator Temperature [K]",
        "clad_temp_K": "Clad Temperature [K]",
        "mod_density_g_cm3": "Moderator Density [g/cm^3]",
        "boron_ppm": "Boron concentration in Moderator [ppm]",
        "burnup_MWd_kg": "Burnup [MWd/kg]",
    }

    for param, ylabel in parameters.items():
        if param in data:
            plt.figure(figsize=(12, 6))
            plt.plot(
                time[1:],
                data[param][:-1],
                marker="o",
                linestyle="None",
                color="black",
                label=param,
            )
            plt.xlabel("Time [d]")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} vs Time")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, f"{param}.png"), dpi=300)
            plt.close()
