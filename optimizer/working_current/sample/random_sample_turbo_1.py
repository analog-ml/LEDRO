import numpy as np
from collections import OrderedDict
import yaml
import yaml.constructor

from turbo.turbo import Turbo1
from turbo.turbo import TurboM
import numpy as np
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
import globalsy
from spectre_simulator.spectre.meas_script.fully_differential_folded_cascode_meas_man import *


import sys
from loguru import logger

logger.remove()
log_level = "DEBUG"
# log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
# Custom format string
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level>| "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Clear default logger
logger.remove()

# Log to stdout
logger.add(sys.stdout, format=log_format, level="INFO")

# Log to file with rotation and retention
logger.add(
    "logs/turbo1_optimizer.log",
    format=log_format,
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
)


np.random.seed(1299)
region_mapping = {
    0: "cut-off",
    1: "triode",
    2: "saturation",
    3: "sub-threshold",
    4: "breakdown",
}


class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor("tag:yaml.org,2002:map", type(self).construct_yaml_map)
        self.add_constructor("tag:yaml.org,2002:omap", type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "expected a mapping node, but found %s" % node.id,
                node.start_mark,
            )

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


# Define the ranges
nA1_range = (2.5e-7, 4.5e-7)
nB1_range = (3, 5)
nA2_range = (2.8e-7, 4.8e-7)
nB2_range = (2, 4)
nA3_range = (1.8e-7, 3.8e-7)
nB3_range = (3, 5)
nA4_range = (4.2e-7, 6.2e-7)
nB4_range = (2, 4)
nA5_range = (3.2e-7, 5.2e-7)
nB5_range = (4, 6)
nA6_range = (4.5e-7, 6.5e-7)
nB6_range = (3, 5)
nA7_range = (10e-9, 400e-9)
nB7_range = (1, 7)
nA8_range = (10e-9, 400e-9)
nB8_range = (1, 7)
nA9_range = (10e-9, 400e-9)
nB9_range = (1, 7)
vbiasp0_range = (0, 0.8)
vbiasp1_range = (0.35, 0.55)
vbiasp2_range = (0.25, 0.45)
vbiasn0_range = (0.4, 0.6)
vbiasn1_range = (0.2, 0.4)
vbiasn2_range = (0.45, 0.65)
cc_range = (1e-15, 1e-11)
vcm = 0.40
vdd = 0.8
tempc = 27

lb = np.array(
    [
        nA1_range[0],
        nB1_range[0],
        nA2_range[0],
        nB2_range[0],
        nA3_range[0],
        nB3_range[0],
        nA4_range[0],
        nB4_range[0],
        nA5_range[0],
        nB5_range[0],
        nA6_range[0],
        nB6_range[0],
        # nA7_range[0],
        # nB7_range[0],
        # nA8_range[0],
        # nB8_range[0],
        # nA9_range[0],
        # nB9_range[0],
        # vbiasp0_range[0],
        vbiasp1_range[0],
        vbiasp2_range[0],
        vbiasn0_range[0],
        vbiasn1_range[0],
        vbiasn2_range[0],
        # cc_range[0],
    ]
)
ub = np.array(
    [
        nA1_range[1],
        nB1_range[1],
        nA2_range[1],
        nB2_range[1],
        nA3_range[1],
        nB3_range[1],
        nA4_range[1],
        nB4_range[1],
        nA5_range[1],
        nB5_range[1],
        nA6_range[1],
        nB6_range[1],
        # nA7_range[1],
        # nB7_range[1],
        # nA8_range[1],
        # nB8_range[1],
        # nA9_range[1],
        # nB9_range[1],
        # vbiasp0_range[1],
        vbiasp1_range[1],
        vbiasp2_range[1],
        vbiasn0_range[1],
        vbiasn1_range[1],
        vbiasn2_range[1],
        # cc_range[1]
    ]
)
# Get a random sample

CIR_YAML = (
    "spectre_simulator/spectre/specs_list_read/fully_differential_folded_cascode.yaml"
)
with open(CIR_YAML, "r") as f:
    yaml_data = yaml.load(f, OrderedDictYAMLLoader)
f.close()
params = yaml_data["params"]
specs = yaml_data["target_spec"]
specs_ideal = []
for spec in list(specs.values()):
    specs_ideal.append(spec)
specs_ideal = np.array(specs_ideal)
params_id = list(params.keys())
specs_id = list(specs.keys())

# import ipdb; ipdb.set_trace()


class Levy:
    def __init__(
        self,
        dim,
        params_id: list[str],
        specs_id: list[str],
        specs_ideal: list[float],
        vcm: float,
        vdd: float,
        tempc: float,
        ub: np.ndarray,
        lb: np.ndarray,
    ):
        """
        :param dim: Dimension of the problem. For analog sizing with fully differential folded cascode opamp, this is 17.
        :param params_id: List of parameter names
        :param specs_id: List of specification names
        :param specs_ideal: Ideal values for specifications
        :param vcm: Common mode voltage
        :param vdd: Supply voltage
        :param tempc: Temperature in Celsius
        :param ub: Upper bounds for parameters
        :param lb: Lower bounds for parameters
        """

        self.dim = dim
        self.params_id = params_id  # parameter names
        self.specs_id = specs_id  # specification names
        self.specs_ideal = specs_ideal
        self.vcm = vcm
        self.vdd = vdd
        self.tempc = tempc
        self.ub = ub
        self.lb = lb

    def lookup(self, spec: list[float], goal_spec: list[float]) -> np.ndarray:
        """
        Normalize the specifications based on their ideal values.
        This function normalizes the specifications by calculating the relative difference
        between the current specification values and the ideal values.
        The normalization is done as per the formula:
        (spec - goal_spec) / (goal_spec + spec)
        This is a common approach to normalize specifications in analog design optimization.
        Refer to the subsection II. Figure of Merit in the paper for details.

        :param spec: Current specification values
        :param goal_spec: Ideal specification values
        :return: Normalized specifications
        """
        # assert isinstance(spec, list)
        # assert isinstance(goal_spec, list)

        goal_spec = [float(e) for e in goal_spec]
        spec = [float(e) for e in spec]
        spec = np.array(spec)
        goal_spec = np.array(goal_spec)

        norm_spec = (spec - goal_spec) / (np.abs(goal_spec) + np.abs(spec))
        # (spec-goal_spec)/(goal_spec+spec)
        return norm_spec

    def reward(self, spec: list[float], goal_spec: list[float], specs_id: list[str]):
        """
        Calculate the reward based on the specifications and their ideal values.
        :param spec: Current specification values
        :param goal_spec: Ideal specification values
        :param specs_id: List of specification names
        :return: Reward value
        """
        # assert isinstance(spec, list)
        # assert isinstance(goal_spec, list)
        # assert isinstance(specs_id, list)

        if len(spec) != len(goal_spec) or len(spec) != len(specs_id):
            raise ValueError("spec, goal_spec, and specs_id must have the same length")

        norm_specs = self.lookup(spec, goal_spec)

        # pay attention to reward calculation, this is not quite the reward function in RL
        # but rather a penalty value for the optimization process
        reward = 0
        for i, rel_spec in enumerate(norm_specs):
            # For power,  smaller is better
            # For gain, larger (compared to the target/goal) is better
            # For other specs (pm, ugbw, etc.), smaller is better
            if specs_id[i] == "power" and rel_spec > 0:
                reward += np.abs(rel_spec)  # /10
            elif specs_id[i] == "gain" and rel_spec < 0:
                reward += 3 * np.abs(rel_spec)  # /10
            elif specs_id[i] != "power" and rel_spec < 0:
                reward += np.abs(rel_spec)

        logger.debug(
            f"reward: {reward:.3g} for specs: {spec} and ideal specs: {goal_spec}"
        )
        return reward  ###updated

    def __call__(self, x):
        """
        :param x: A numpy array of shape (dim,) representing the parameters to evaluate.
        :return: A float value representing the reward for the given parameters.
        """
        reward, _ = self.evaluate(x)
        return reward

    def evaluate(self, x):
        """
        Evaluate the objective function for a given sample x.
        :param x: A numpy array of shape (dim,) representing the parameters to evaluate.
        :return: A float value representing the reward for the given parameters.
        """
        # assert isinstance(x, np.ndarray)
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        CIR_YAML = "spectre_simulator/spectre/specs_list_read/fully_differential_folded_cascode.yaml"
        sim_env = OpampMeasMan(CIR_YAML)
        sample = x

        # Round the values of Nfins (number of fins) to the nearest integer
        sample[1] = round(sample[1])
        sample[3] = round(sample[3])
        sample[5] = round(sample[5])
        sample[7] = round(sample[7])
        sample[9] = round(sample[9])
        sample[11] = round(sample[11])
        # sample[13] = round(sample[13])
        # sample[15] = round(sample[15])
        # sample[17] = round(sample[17])
        sample = np.append(sample, self.vcm)
        sample = np.append(sample, self.vdd)
        sample = np.append(sample, self.tempc)

        param_val = [OrderedDict(list(zip(self.params_id, sample)))]

        cur_specs = OrderedDict(
            sorted(sim_env.evaluate(param_val)[0][1].items(), key=lambda k: k[0])
        )
        dict1 = OrderedDict(list(cur_specs.items())[:-5])  # all the original
        cur_specs = np.array(list(dict1.values()))[:-1]
        dummy = cur_specs[0]
        cur_specs[0] = cur_specs[1]
        cur_specs[1] = dummy

        reward1 = self.reward(cur_specs, self.specs_ideal, self.specs_id)
        logger.info(f"reward: {reward1:.3g}")
        return reward1, cur_specs


f = Levy(17, params_id, specs_id, specs_ideal, vcm, vdd, tempc, ub, lb)

turbo1 = Turbo1(
    f=f,  # Handle to objective function
    lb=lb,  # Numpy array specifying lower bounds
    ub=ub,  # Numpy array specifying upper bounds
    n_init=200,  # Number of initial bounds from an Latin hypercube design
    max_evals=1000,  # Maximum number of evaluations
    batch_size=128,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=100,  # Number of steps of ADAM to learn the hypers
    min_cuda=10.40,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float32",  # float64 or float32
)

turbo1.optimize()

X = turbo1.X  # Evaluated points
fX = turbo1.fX  # Observed values
ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]

np.save("X.npy", X)
np.save("fX", fX)
np.save("fSpec", turbo1.infoX)

print("Optimization completed.")
print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, x_best))
