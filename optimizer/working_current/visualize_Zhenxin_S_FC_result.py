import numpy as np
from collections import OrderedDict


# =====================
# Load optimization results
# =====================
# Assuming fX and X are numpy arrays saved in the current directory
# If they are not, adjust the paths accordingly
fX = np.load("fX.npy")
fSpec = np.load("fSpec.npy")
X = np.load("X.npy")

ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]


print("Optimization completed.")
print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, x_best))

# Load simulation environment
from spectre_simulator.spectre.meas_script.fully_differential_folded_cascode_meas_man import *

CIR_YAML = (
    "spectre_simulator/spectre/specs_list_read/Zhenxin_S_FC.yaml"
)
sim_env = OpampMeasMan(CIR_YAML)


def obtain_spec(x):
    sample = x
    # sample[1] = round(sample[1])
    # sample[3] = round(sample[3])
    # sample[5] = round(sample[5])
    # sample[7] = round(sample[7])
    # sample[9] = round(sample[9])
    # sample[11] = round(sample[11])
    # # sample[13] = round(sample[13])
    # # sample[15] = round(sample[15])
    # # sample[17] = round(sample[17])
    # sample = np.append(sample, 0.4)
    # sample = np.append(sample, 0.8)
    # sample = np.append(sample, 27)

    params_id = [
            "w_m12",
            "w_m3", 
            "w_m45",
            "w_m67", 
            "w_m89", 
            "w_m1011", 
            "vbp1",
            "vbp2",
            "vbn1",
            "vbn2",
            "cc"
    ]
    param_val = [OrderedDict(list(zip(params_id, sample)))]
    cur_specs = OrderedDict(
        sorted(sim_env.evaluate(param_val)[0][1].items(), key=lambda k: k[0])
    )
    # print("Current specs:")
    cur_specs = dict(cur_specs)  # Convert OrderedDict to dict for easier printing
    # print(cur_specs)
    return cur_specs


print("Obtaining specs for the best sample...")
best_specs = obtain_spec(x_best)
print("Best specs obtained:", best_specs)

import matplotlib.pyplot as plt


def plot_optimization_results(X, fX, fSpec):
    gains = [fSpec[i][0] for i in range(len(fSpec))]
    funities = [fSpec[i][1] for i in range(len(fSpec))]

    indices = np.arange(len(X))

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(indices, gains, label="Gain")
    axs[0].set_ylabel("Gain")
    axs[0].set_title("Gain vs. Iteration")
    axs[0].legend()

    axs[1].plot(indices, funities, label="Funity", color="orange")
    axs[1].set_ylabel("Funity")
    axs[1].set_title("Funity vs. Iteration")
    axs[1].legend()

    axs[2].plot(indices, -1.0 * fX, label="FoM", color="green")
    axs[2].set_ylabel("FoM")
    axs[2].set_title("FoM vs. Iteration")
    axs[2].set_xlabel("Iteration")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("optimization_results.png")
    # plt.show()


plot_optimization_results(X, fX, fSpec)
from scipy.signal import savgol_filter  # <-- Add this import


def plot_optimization_results2(
    X, fX, fSpec, smooth=True, window_length=51, polyorder=3
):
    gains = [fSpec[i][0] for i in range(len(fSpec))]
    funities = [fSpec[i][1] for i in range(len(fSpec))]
    fom = -1.0 * fX.flatten()
    indices = np.arange(len(X))

    # Apply smoothing if requested and data is long enough
    if smooth:
        if len(gains) >= window_length:
            gains_smooth = savgol_filter(gains, window_length, polyorder)
        else:
            gains_smooth = gains
        if len(funities) >= window_length:
            funities_smooth = savgol_filter(funities, window_length, polyorder)
        else:
            funities_smooth = funities
        if len(fom) >= window_length:
            fom_smooth = savgol_filter(fom, window_length, polyorder)
        else:
            fom_smooth = fom
    else:
        gains_smooth = gains
        funities_smooth = funities
        fom_smooth = fom

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(indices, gains_smooth, label="Gain (smoothed)")
    axs[0].set_ylabel("Gain")
    axs[0].set_title("Gain vs. Iteration")
    axs[0].legend()

    axs[1].plot(indices, funities_smooth, label="UGBW (smoothed)", color="orange")
    axs[1].set_ylabel("UGBW")
    axs[1].set_title("UGBW vs. Iteration")
    axs[1].legend()

    axs[2].plot(indices, fom_smooth, label="FoM (smoothed)", color="green")
    axs[2].set_ylabel("FoM")
    axs[2].set_title("FoM vs. Iteration")
    axs[2].set_xlabel("Iteration")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("optimization_results_smoothed.png")
    # plt.show()


plot_optimization_results2(X, fX, fSpec, window_length=100)
