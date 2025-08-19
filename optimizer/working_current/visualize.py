import numpy as np
from collections import OrderedDict


# =====================
# Load optimization results
# =====================
# Assuming fX and X are numpy arrays saved in the current directory
# If they are not, adjust the paths accordingly
fX = np.load("fX.npy")
X = np.load("X.npy")

ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]

np.save("X.npy", X)
np.save("fX.npy", fX)

print("Optimization completed.")
print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, x_best))


# Load simulation environment
from spectre_simulator.spectre.meas_script.fully_differential_folded_cascode_meas_man import *

CIR_YAML = (
    "spectre_simulator/spectre/specs_list_read/fully_differential_folded_cascode.yaml"
)
sim_env = OpampMeasMan(CIR_YAML)


def obtain_spec(x):
    sample = x
    sample[1] = round(sample[1])
    sample[3] = round(sample[3])
    sample[5] = round(sample[5])
    sample[7] = round(sample[7])
    sample[9] = round(sample[9])
    sample[11] = round(sample[11])
    # sample[13] = round(sample[13])
    # sample[15] = round(sample[15])
    # sample[17] = round(sample[17])
    sample = np.append(sample, 0.4)
    sample = np.append(sample, 0.8)
    sample = np.append(sample, 27)

    params_id = [
        "nA1",
        "nB1",
        "nA2",
        "nB2",
        "nA3",
        "nB3",
        "nA4",
        "nB4",
        "nA5",
        "nB5",
        "nA6",
        "nB6",
        "vbiasp1",
        "vbiasp2",
        "vbiasn0",
        "vbiasn1",
        "vbiasn2",
        "vcm",
        "vdd",
        "tempc",
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


def plot_optimization_results(X, fX):
    gains = []
    funities = []
    for x in X:
        specs = obtain_spec(x)
        gains.append(specs["gain"])
        funities.append(specs["funity"])

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

    axs[2].plot(indices, -1.0 * fX, label="fX", color="green")
    axs[2].set_ylabel("FoM")
    axs[2].set_title("FoM vs. Iteration")
    axs[2].set_xlabel("Iteration")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("optimization_results.png")
    # plt.show()


plot_optimization_results(X, fX)
