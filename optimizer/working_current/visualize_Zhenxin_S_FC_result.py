import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.signal import savgol_filter  # <-- Add this import
import pickle

from torch.utils.tensorboard import SummaryWriter
import numpy as np

# =====================
# Load optimization results
# =====================
# Assuming fX and X are numpy arrays saved in the current directory
# If they are not, adjust the paths accordingly
fX = np.load("fX.npy")
# fSpec = np.load("fSpec.npy")
X = np.load("X.npy")
with open("simulation.dat", "rb") as f:
    loaded_dict = pickle.load(f)
    fSpec = [d["cur_specs"] for d in loaded_dict]
    original_FoM = [d["original_reward"] for d in loaded_dict]

original_FoM = np.array(original_FoM)
# print (fSpec)
# exit()
# print (fX)
# print(original_FoM)
# exit()

ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]


print("Optimization completed.")
print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, x_best))

# Load simulation environment
from spectre_simulator.spectre.meas_script.fully_differential_folded_cascode_meas_man import *

CIR_YAML = "spectre_simulator/spectre/specs_list_read/Zhenxin_S_FC.yaml"
sim_env = OpampMeasMan(CIR_YAML)


def obtain_spec(x):
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
        "cc",
    ]
    param_val = [OrderedDict(list(zip(params_id, x)))]
    cur_specs = OrderedDict(
        sorted(sim_env.evaluate(param_val)[0][1].items(), key=lambda k: k[0])
    )
    cur_specs = dict(cur_specs)
    return cur_specs


print("Obtaining specs for the best sample...")
best_specs = obtain_spec(x_best)
print("Best specs obtained:", best_specs)


tb_name = input("Enter tensorboard log name (e.g., runs/exp1): ")
tb_name = tb_name.strip().replace(" ", "_")
writer = SummaryWriter(tb_name)


def plot_optimization_results(X, fX, fSpec):
    gains = [fSpec[i][0] for i in range(len(fSpec))]
    funities = [fSpec[i][1] for i in range(len(fSpec))]
    pm = [fSpec[i][2] for i in range(len(fSpec))]
    power = [fSpec[i][3] for i in range(len(fSpec))]

    indices = np.arange(len(X))

    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(indices, gains, label="Gain")
    axs[0].set_ylabel("Gain")
    axs[0].set_title("Gain vs. Iteration")
    axs[0].legend()

    axs[1].plot(indices, funities, label="Funity", color="orange")
    axs[1].set_ylabel("Funity")
    axs[1].set_title("Funity vs. Iteration")
    axs[1].legend()

    axs[2].plot(indices, pm, label="PM", color="red")
    axs[2].set_ylabel("PM")
    axs[2].set_title("PM vs. Iteration")
    axs[2].legend()

    axs[3].plot(indices, power, label="Power", color="blue")
    axs[3].set_ylabel("Power")
    axs[3].set_title("Power vs. Iteration")
    axs[3].legend()

    axs[4].plot(indices, -1.0 * fX, label="FoM", color="green")
    axs[4].plot(
        indices, -1.0 * original_FoM, label="FoM-org", color="purple"
    )  # , linestyle='dashed')

    axs[4].set_ylabel("FoM")
    axs[4].set_title("FoM vs. Iteration")
    axs[4].set_xlabel("Iteration")
    axs[4].legend()

    plt.tight_layout()
    plt.savefig("optimization_results.png")

    for i in range(len(gains)):
        writer.add_scalar("Gain", gains[i], i)
        writer.add_scalar("UGBW", funities[i], i)
        writer.add_scalar("PM", pm[i], i)
        writer.add_scalar("Power", power[i], i)
        writer.add_scalar("FoM", -1.0 * fX[i], i)
        writer.add_scalar("FoM-org", -1.0 * original_FoM[i], i)


def plot_optimization_results2(
    X, fX, fSpec, smooth=True, window_length=51, polyorder=3
):
    gains = [fSpec[i][0] for i in range(len(fSpec))]
    funities = [fSpec[i][1] for i in range(len(fSpec))]
    pm = [fSpec[i][2] for i in range(len(fSpec))]
    power = [fSpec[i][3] for i in range(len(fSpec))]

    fom = -1.0 * fX.flatten()
    fom_org = -1.0 * original_FoM.flatten()
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

        if len(pm) >= window_length:
            pm_smooth = savgol_filter(pm, window_length, polyorder)
        else:
            pm_smooth = pm

        if len(power) >= window_length:
            power_smooth = savgol_filter(power, window_length, polyorder)
        else:
            power_smooth = power

        if len(fom) >= window_length:
            fom_smooth = savgol_filter(fom, window_length, polyorder)
        else:
            fom_smooth = fom

        if len(fom_org) >= window_length:
            fom_org_smooth = savgol_filter(fom_org, window_length, polyorder)
        else:
            fom_org_smooth = fom_org
    else:
        gains_smooth = gains
        funities_smooth = funities
        fom_smooth = fom
        fom_org_smooth = fom_org

    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(indices, gains_smooth, label="Gain (smoothed)")
    axs[0].set_ylabel("Gain")
    axs[0].set_title("Gain vs. Iteration")
    axs[0].legend()

    axs[1].plot(indices, funities_smooth, label="UGBW (smoothed)", color="orange")
    axs[1].set_ylabel("UGBW")
    axs[1].set_title("UGBW vs. Iteration")
    axs[1].legend()

    axs[2].plot(indices, pm_smooth, label="PM (smoothed)", color="red")
    axs[2].set_ylabel("PM")
    axs[2].set_title("PM vs. Iteration")
    axs[2].legend()

    axs[3].plot(indices, power_smooth, label="Power (smooth)", color="blue")
    axs[3].set_ylabel("Power")
    axs[3].set_title("Power vs. Iteration")
    axs[3].legend()

    axs[4].plot(indices, fom_smooth, label="FoM (smoothed)", color="green")
    axs[4].plot(
        indices, fom_org_smooth, label="FoM-org (smoothed)", color="purple"
    )  # , linestyle='dashed')
    axs[4].set_ylabel("FoM")
    axs[4].set_title("FoM vs. Iteration")
    axs[4].set_xlabel("Iteration")
    axs[4].legend()

    plt.tight_layout()
    plt.savefig("optimization_results_smoothed.png")
    # plt.show()


plot_optimization_results(X, fX, fSpec)
plot_optimization_results2(X, fX, fSpec, window_length=100)
