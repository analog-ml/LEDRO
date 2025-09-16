import os
import os
import subprocess
import yaml
import importlib
import random
import numpy as np

import shutil
import sys
from spectre_simulator.util.core import IDEncoder, Design
from jinja2 import Environment, FileSystemLoader
from multiprocessing.dummy import Pool as ThreadPool
from loguru import logger

debug = False

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
    "logs/core.log",
    format=log_format,
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
)


def get_config_info():
    # TODO
    config_info = dict()
    base_tmp_dir = os.environ.get("BASE_TMP_DIR", None)
    if not base_tmp_dir:
        raise EnvironmentError("BASE_TMP_DIR is not set in environment variables")
    else:
        config_info["BASE_TMP_DIR"] = base_tmp_dir

    return config_info


class SpectreWrapper(object):

    def __init__(self, tb_dict):
        """
        This Wrapper handles one netlist at a time, meaning that if there are multiple test benches
        for characterizations multiple instances of this class needs to be created

        :param netlist_loc: the template netlist used for circuit simulation
        """

        # suppose we have a config_info = {'section':'model_lib'}
        # config_info also contains BASE_TMP_DIR (location for storing simulation netlist/results)
        # implement get_config_info() later

        netlist_loc = tb_dict["netlist_template"]
        # print(netlist_loc)
        if not os.path.isabs(netlist_loc):
            netlist_loc = os.path.abspath(netlist_loc)
        pp_module = importlib.import_module(tb_dict["tb_module"])
        self.pp_class = getattr(pp_module, tb_dict["tb_class"])
        self.post_process = getattr(self.pp_class, tb_dict["post_process_function"])
        self.tb_params = tb_dict["tb_params"]

        self.config_info = get_config_info()

        self.root_dir = self.config_info["BASE_TMP_DIR"]
        self.num_process = self.config_info.get("NUM_PROCESS", 1)

        _, dsn_netlist_fname = os.path.split(netlist_loc)
        self.base_design_name = os.path.splitext(dsn_netlist_fname)[0] + str(
            random.randint(0, 10000)
        )
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        os.makedirs(self.gen_dir, exist_ok=True)

        file_loader = FileSystemLoader(os.path.dirname(netlist_loc))
        self.jinja_env = Environment(loader=file_loader)
        self.template = self.jinja_env.get_template(dsn_netlist_fname)

    def _get_design_name(self, state):
        """
        Creates a unique identifier fname based on the state
        :param state:
        :return:
        """
        fname = self.base_design_name
        for value in state.values():
            if value <= 2e-13:  # cap
                x = value * 1e14
                fname += "_" + str(round(x, 2))
            elif value <= 1e-6:  # lengths
                x = value * 1e7
                fname += "_" + str(round(x, 2))
            else:
                fname += "_" + str(round(value, 2))
        return fname

    def _create_design(self, state, new_fname):
        design_folder = os.path.join(self.gen_dir, new_fname)

        logger.debug(f"Creating a new design with {state=} ")
        state['design_path'] = design_folder
        output = self.template.render(**state)

        os.makedirs(design_folder, exist_ok=True)
        fpath = os.path.join(design_folder, new_fname + ".cir")
        with open(fpath, "w") as f:
            f.write(output)
            f.close()
        logger.debug(f"Creating a new design at {fpath=} ")
        return design_folder, fpath

    def _simulate(self, fpath):
        command = ["ngspice", "-b", fpath]

        exit_code = subprocess.call(
            command,
            cwd=os.path.dirname(fpath),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        info = 0
        if debug:
            print(command)
            print(fpath)
        if exit_code % 256:
            info = 1  # this means an error has occurred
        return info

    def _create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        if debug:
            print("state", state)
            print("verbose", verbose)
        if dsn_name == None:
            dsn_name = self._get_design_name(state)
        else:
            dsn_name = str(dsn_name)
        if verbose:
            print("dsn_name", dsn_name)

        design_folder, fpath = self._create_design(state, dsn_name)
        info = self._simulate(fpath)
        results = self._parse_result(design_folder)
        if self.post_process:
            specs = self.post_process(results, self.tb_params)
            # shutil.rmtree(design_folder)
            #    print("design_folder", design_folder)
            return state, specs, info

        specs = results
        return state, specs, info


    def parse_output(self, output_path):

        ac_fname = os.path.join(output_path, "ac.csv")
        dc_fname = os.path.join(output_path, "dc.csv")

        if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
            print("ac/dc file doesn't exist: %s" % output_path)

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout_real = ac_raw_outputs[:, 1]
        vout_imag = ac_raw_outputs[:, 2]
        vout = vout_real + 1j * vout_imag
        ibias = -dc_raw_outputs[1]

        return freq, vout, ibias
    def _parse_result(self, design_folder):

        logger.debug(f"Parsing results from design folder: {design_folder}")
        freq, vout, ibias = self.parse_output(design_folder)

        gain = getattr(self.pp_class, "find_dc_gain")(vout)
        ugbw, valid = getattr(self.pp_class, "find_ugbw")(freq, vout)
        phm = getattr(self.pp_class, "find_phm")(freq, vout)

        if not os.path.exists(design_folder):
            raise FileNotFoundError(f"Design folder {design_folder} does not exist")

        metrics = {}
        metrics["ugbw"] = ugbw
        metrics["pm"] = phm
        metrics["ac_gain"] = gain
        metrics["power"] = ibias
        metrics["valid"] = valid
        res = {"df": None, "metrics": metrics}
        logger.info(f"Metrics: {metrics}")

        return res

    def run(self, states, design_names=None, verbose=False):
        # TODO: Use asyncio to instantiate multiple jobs for running parallel sims
        """

        :param states:
        :param design_names: if None default design name will be used, otherwise the given design name will be used
        :param verbose: If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value), info: int)]
        """
        pool = ThreadPool(processes=self.num_process)
        arg_list = [
            (state, dsn_name, verbose)
            for (state, dsn_name) in zip(states, design_names)
        ]
        specs = pool.starmap(self._create_design_and_simulate, arg_list)
        pool.close()
        return specs

    def return_path(self):
        # print(self.gen_dir)
        return self.gen_dir


class EvaluationEngine(object):

    def __init__(self, yaml_fname):

        self.design_specs_fname = yaml_fname
        with open(yaml_fname, "r") as f:
            self.ver_specs = yaml.load(f, Loader=yaml.Loader)
        f.close()

        self.spec_range = self.ver_specs["spec_range"]
        # params are interfaced using the index instead of the actual value
        params = self.ver_specs["params"]

        self.params_vec = {}
        self.search_space_size = 1
        for key, value in params.items():
            self.params_vec[key] = np.arange(value[0], value[1], value[2]).tolist()
            self.search_space_size = self.search_space_size * len(self.params_vec[key])

        # minimum and maximum of each parameter
        # params_vec contains the acutal numbers but min and max should be the indices
        self.params_min = [0] * len(self.params_vec)
        self.params_max = []
        for val in self.params_vec.values():
            self.params_max.append(len(val) - 1)

        self.id_encoder = IDEncoder(self.params_vec)
        self.measurement_specs = self.ver_specs["measurement"]
        tbs = self.measurement_specs["testbenches"]
        self.netlist_module_dict = {}
        for tb_kw, tb_val in tbs.items():
            self.netlist_module_dict[tb_kw] = SpectreWrapper(tb_val)

    @property
    def num_params(self):
        return len(self.params_vec)

    def generate_data_set(self, n=1, debug=False, parallel_config=None):
        """
        :param n:
        :return: a list of n Design objects with populated attributes (i.e. cost, specs, id)
        """
        valid_designs, tried_designs = [], []
        nvalid_designs = 0

        useless_iter_count = 0
        while len(valid_designs) < n:
            design = {}
            for key, vec in self.params_vec.items():
                rand_idx = random.randrange(len(vec))
                design[key] = rand_idx
            design = Design(self.spec_range, self.id_encoder, list(design.values()))
            if design in tried_designs:
                if useless_iter_count > n * 5:
                    raise ValueError(
                        "Random selection of a fraction of search space did not "
                        "result in {} number of valid designs".format(n)
                    )
                useless_iter_count += 1
                continue
            design_result = self.evaluate([design], debug=debug)[0]
            if design_result["valid"]:
                design.cost = design_result["cost"]
                for key in design.specs.keys():
                    design.specs[key] = design_result[key]
                valid_designs.append(design)
            else:
                nvalid_designs += 1
            tried_designs.append(design)
            print(len(valid_designs))
            print("not valid designs:" + str(nvalid_designs))
        return valid_designs[:n]

    def evaluate(self, design_list, debug=True, parallel_config=None):
        """
        serial implementation of evaluate (not parallel)
        :param design_list:
        :return: a list of processed_results that the algorithm cares about, keywords should include
        cost and spec keywords with one scalar number as the value for each
        """
        results = []
        if len(design_list) > 1:
            for design in design_list:
                try:
                    result = self._evaluate(design, parallel_config=parallel_config)
                    # result['valid'] = True
                except Exception as e:
                    if debug:
                        raise e
                    result = {"valid": False}
                    print(getattr(e, "message", str(e)))

                results.append(result)
        else:
            try:
                netlist_name, netlist_module = list(self.netlist_module_dict.items())[0]
                result = netlist_module._create_design_and_simulate(design_list[0])
            except Exception as e:
                if debug:
                    raise e
                result = {"valid": False}
                print(getattr(e, "message", str(e)))
            results.append(result)
        return_loc = netlist_module.return_path()
        # print(return_loc)
        shutil.rmtree(return_loc)
        logger.info(f"[x] simulated {results=}")
        return results

    def _evaluate(self, design, parallel_config):
        state_dict = dict()
        for i, key in enumerate(self.params_vec.keys()):
            state_dict[key] = self.params_vec[key][design[i]]
        state = [state_dict]
        dsn_names = [design.id]
        print(state_dict)
        results = {}
        for netlist_name, netlist_module in self.netlist_module_dict.items():
            results[netlist_name] = netlist_module.create_design_and_simulate(
                state, dsn_names
            )

        specs_dict = self.get_specs(results, self.measurement_specs["meas_params"])
        print(specs_dict)
        specs_dict["cost"] = self.cost_fun(specs_dict)
        return specs_dict

    def find_worst(self, spec_nums, spec_kwrd, ret_penalty=False):
        if not hasattr(spec_nums, "__iter__"):
            spec_nums = [spec_nums]

        penalties = self.compute_penalty(spec_nums, spec_kwrd)
        worst_penalty = max(penalties)
        worst_idx = penalties.index(worst_penalty)
        if ret_penalty:
            return spec_nums[worst_idx], worst_penalty
        else:
            return spec_nums[worst_idx]

    def cost_fun(self, specs_dict):
        """
        :param design: a list containing relative indices according to yaml file
        :param verbose:
        :return:
        """
        cost = 0
        for spec in self.spec_range.keys():
            penalty = self.compute_penalty(specs_dict[spec], spec)[0]
            cost += penalty

        return cost

    def get_specs(self, results, params):
        """
        converts results from different tbs to a single dictionary summarizing everything
        :param results:
        :return:
        """
        raise NotImplementedError

    def compute_penalty(self, spec_nums, spec_kwrd):
        # use self.spec_range[spec_kwrd] to get the min, max, and weight
        raise NotImplementedError


from collections import OrderedDict
import yaml
import yaml.constructor

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


def main():
    # testing the cs amp functionality with Jinja2
    CIR_YAML = "/home/pham/code/analog-ml/LEDRO/optimizer/working_current/spectre_simulator/spectre/specs_list_read/Zhenxin_S_FC.yaml"
    with open(CIR_YAML, "r") as f:
        yaml_data = yaml.load(f, OrderedDictYAMLLoader)

    opamp_env = SpectreWrapper(tb_dict=yaml_data["measurement"]["testbenches"]["ac_dc"])

    keys = [
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
    vals = [242.84323390575366,
    133.8078576088182,
    292.64831036522917,
    198.92456090951285,
    139.80927503463723,
    131.13102449401783,
    0.6568611016690267,
    0.17811360700059536,
    0.686108575948138,
    0.3053737857576733,
    1.0]
    state = dict(zip(keys, vals))

    state, specs, info = opamp_env._create_design_and_simulate(state)
    logger.info(state)
    logger.info(specs)

    from sample.random_sample_turbo_1 import Levy
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
        cc_range[0]
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
        cc_range[1]
        ]
    )

    CIR_YAML = (
        "spectre_simulator/spectre/specs_list_read/Zhenxin_S_FC.yaml"
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
    # f = Levy(17, params_id, specs_id, specs_ideal, vcm, vdd, tempc, ub, lb)

    # # print (f())
    # x = list(state.values())[:-3]
    # x = np.array(x).reshape(-1)
    # print (f.evaluate(x))

if __name__ == "__main__":
    main()
