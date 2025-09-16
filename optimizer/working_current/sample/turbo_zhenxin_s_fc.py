import numpy as np
import yaml
import yaml.constructor
import sys
import os
import numpy as np
import pickle

from loguru import logger
from collections import OrderedDict
from turbo.turbo import Turbo1
from spectre_simulator.spectre.meas_script.Zhenxin_S_FC_meas import *

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
logger.add(sys.stdout, format=log_format, level="DEBUG")

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


w_m12_range = [130, 100000]
w_m3_range = [50, 100000]
w_m45_range = [130, 100000]
w_m67_range = [80, 100000]
w_m89_range = [50, 100000]
w_m1011_range = [50, 100000]
vbp1_range = [0.0001, 1.1]
vbp2_range = [0.0001, 1.1]
vbn1_range = [0.0001, 1.1]
vbn2_range = [0.0001, 1.1]
cc_range = [0.01, 10]

lb = np.array(
    [
        w_m12_range[0],
        w_m3_range[0],
        w_m45_range[0],
        w_m67_range[0],
        w_m89_range[0],
        w_m1011_range[0],
        vbp1_range[0],
        vbp2_range[0],
        vbn1_range[0],
        vbn2_range[0],
        cc_range[0],
    ]
)
ub = np.array(
    [
        w_m12_range[1],
        w_m3_range[1],
        w_m45_range[1],
        w_m67_range[1],
        w_m89_range[1],
        w_m1011_range[1],
        vbp1_range[1],
        vbp2_range[1],
        vbn1_range[1],
        vbn2_range[1],
        cc_range[1],
    ]
)


# Get a random sample
CIR_YAML = "spectre_simulator/spectre/specs_list_read/Zhenxin_S_FC.yaml"
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


class Levy:
    def __init__(
        self,
        dim,
        params_id: list[str],
        specs_id: list[str],
        specs_ideal: list[float],
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
        self.ub = ub
        self.lb = lb

        self.reward_idx = 1
        self.last_change = 0
        self.last_params = None

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
        logger.debug(f"current spec: {spec}")
        if len(spec) != len(goal_spec) or len(spec) != len(specs_id):
            raise ValueError("spec, goal_spec, and specs_id must have the same length")

        norm_specs = self.lookup(spec, goal_spec)

        # pay attention to reward calculation, this is not quite the reward function in RL
        # but rather a penalty value for the optimization process

        def calc_reward(w):
            # w = [1, 1, 1, 1] # weights for gain, power, pm, ugbw

            reward = 0
            for i, rel_spec in enumerate(norm_specs):
                if specs_id[i] == "power" and rel_spec > 0:
                    reward += w[1] * np.abs(rel_spec)
                elif specs_id[i] == "gain" and rel_spec < 0:
                    reward += w[0] * np.abs(rel_spec)
                elif specs_id[i] == "funity" and rel_spec < 0:
                    reward += w[-1] * np.abs(rel_spec)
                elif specs_id[i] == "pm" and rel_spec < 0:
                    reward += w[-2] * np.abs(rel_spec)
            return reward

        self.ret_reward_0 = calc_reward([1, 1, 1, 1])
        self.ret_reward_1 = calc_reward([0.22, 0.15, 0.13, 0.50])
        self.ret_reward_2 = calc_reward([0.34, 0.24, 0.12, 0.30])
        self.ret_reward_3 = calc_reward([0.40, 0.30, 0.15, 0.15])
        self.ret_reward_4 = calc_reward([0.30, 0.40, 0.20, 0.10])
        self.ret_reward_5 = calc_reward([0.22, 0.30, 0.40, 0.08])
        self.ret_reward_6 = calc_reward([0.18, 0.18, 0.24, 0.40])
        self.ret_reward_7 = calc_reward([0.25, 0.25, 0.25, 0.25])

        reward_map = {
            0: self.ret_reward_0,
            1: self.ret_reward_1,
            2: self.ret_reward_2,
            3: self.ret_reward_3,
            4: self.ret_reward_4,
            5: self.ret_reward_5,
            6: self.ret_reward_6,
            7: self.ret_reward_7,
        }

        reward = reward_map[self.reward_idx]

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

        if self.last_change < 40_000:
            self.last_change = self.last_change + 1
        else:
            if self.reward_idx == 1 and self.cur_specs[1] < 12e-3:
                self.reward_idx = 2
                self.last_change = 0

            elif self.reward_idx == 2 and self.cur_specs[0] > 565:
                self.reward_idx = 3
                self.last_change = 0

            elif self.reward_idx == 3 and self.cur_specs[0] > 800:
                self.reward_idx = 4
                self.last_change = 0

            elif self.reward_idx == 4 and self.cur_specs[-1] > 5.0e-6:
                self.reward_idx = 5
                self.last_change = 0

            elif self.reward_idx == 5 and self.cur_specs[-2] > 60.0:
                self.reward_idx = 6
                self.last_change = 0
            elif self.reward_idx == 6 and self.cur_specs[1] < 10e-3:
                self.reward_idx = 7
                self.last_change = 0

        # IMPORTANT: comment out the following lines if you don't want to use adaptive reward function during optimization
        self.reward_idx = 0
        self.last_params = None

        if self.last_params is None:
            self.last_params = np.copy(x)

        if self.reward_idx == 1:
            x[1] = self.last_params[1]
            x[5] = self.last_params[5]

        if self.reward_idx == 2:
            x[0] = self.last_params[0]
            x[1] = self.last_params[1]

        if self.reward_idx == 3:
            x[0] = self.last_params[0]
            x[1] = self.last_params[1]

        if self.reward_idx == 4:
            x[2] = self.last_params[2]
            x[3] = self.last_params[3]
            x[4] = self.last_params[4]

        if self.reward_idx == 5:
            x[2] = self.last_params[2]
            x[3] = self.last_params[3]
            x[4] = self.last_params[4]

        if self.reward_idx == 6:
            x[1] = self.last_params[1]
            x[5] = self.last_params[5]

        if self.reward_idx == 7:
            x[2] = self.last_params[2]
            x[3] = self.last_params[3]
            x[4] = self.last_params[4]

        self.last_params = np.copy(x)

        CIR_YAML = "spectre_simulator/spectre/specs_list_read/Zhenxin_S_FC.yaml"
        sim_env = OpampMeasMan(CIR_YAML)
        sample = x
        param_val = [OrderedDict(list(zip(self.params_id, sample)))]

        cur_specs = OrderedDict(
            sorted(sim_env.evaluate(param_val)[0][1].items(), key=lambda k: k[0])
        )
        dict1 = OrderedDict(list(cur_specs.items())[:-5])  # all the original
        dict3 = OrderedDict(list(cur_specs.items())[-5:-4])  # region
        dict2 = OrderedDict(list(cur_specs.items())[-4:])  # remaining

        dict2_values = list(dict2.values())
        flattened_dict2 = [item for sublist in dict2_values for item in sublist]
        dict2_nparray = np.array(flattened_dict2)

        dict3_values = list(dict3.values())
        flattened_dict3 = [item for sublist in dict3_values for item in sublist]
        dict3_nparray = np.array(flattened_dict3)

        cur_specs = np.array(list(dict1.values()))[:-1]
        dummy = cur_specs[0]
        cur_specs[0] = cur_specs[1]
        cur_specs[1] = dummy
        # f = open("/path/to/optimizer/out1.txt",'a')
        # print("cur_specs", cur_specs, file=f)
        reward1 = self.reward(cur_specs, self.specs_ideal, self.specs_id)
        # f = open("performance+deviceparams.log", "a")
        for ordered_dict in param_val:
            # formatted_items = [
            #     f"{k}: {format(v, '.3g')}" for k, v in ordered_dict.items()
            # ]
            # param_value_str = ", ".join(formatted_items)
            pass
            # print(", ".join(formatted_items), file=f)

        # f.close()

        # print("Device parameters:", param_value_str, file=f)
        # print("reward", format(-reward1, ".3g"), file=f)
        # f.close()
        logger.debug(
            f"Reward: {reward1:.3g} for sample: {sample} with specs: {cur_specs}"
        )
        return reward1, {"cur_specs": cur_specs, "original_reward": self.ret_reward_0}


if __name__ == "__main__":
    if os.path.exists("performance+deviceparams.log"):
        os.remove("performance+deviceparams.log")
    f = Levy(11, params_id, specs_id, specs_ideal, ub, lb)

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
    # np.save("fSpec", turbo1.infoX)
    with open("simulation.dat", "wb") as f:
        pickle.dump(turbo1.infoX, f)

    print("Optimization completed.")
    print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, x_best))
