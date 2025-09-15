import re
import copy
import os
from jinja2 import Environment, FileSystemLoader
import os
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
import yaml
import importlib
import random
import numpy as np
from spectre_simulator.util.core import IDEncoder, Design
from spectre_simulator.spectre.parser import SpectreParser
import IPython
import shutil
import glob
from spmetrics.utils import (
    run_ngspice_simulation,
    setup_ac_simulation,
    setup_op_simulation,
)
from spmetrics.metrics import (
    # AC simulation functions
    compute_bandwidth,
    compute_unity_gain_bandwidth,
    compute_phase_margin,
    compute_ac_gain,
    compute_power,
)

debug = False
import pandas as pd

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
    "logs/core.log",
    format=log_format,
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
)


def get_variable_names(log_path):
    variable_names = []
    with open(log_path, "r") as f:
        in_variables_section = False
        for line in f:
            if line.strip() == "Variables:":
                in_variables_section = True
                continue
            if in_variables_section:
                if line.strip() == "" or line.startswith("Values:"):
                    break
                # Each variable line: <index> <name> <type>
                parts = line.strip().split()
                if len(parts) >= 2:
                    variable_names.append(parts[1])
    return variable_names


def get_values(log_path):
    values = []
    with open(log_path, "r") as f:
        in_values_section = False
        for line in f:
            if line.strip() == "Values:":
                in_values_section = True
                continue
            if in_values_section:
                if line.strip() == "":
                    break
                # Each value line: <index> <value>
                parts = line.strip().split()
                for part in parts:
                    try:
                        values.append(float(part))
                    except ValueError:
                        pass
    return values[1:]  # Skip the first value which is usually an index or header


# Helper function to determine PMOS region
def get_pmos_region(vgs, vds, vth):
    if vgs > vth:
        return 0  # cut-off
    else:
        if vds > vgs - vth:
            return 1  # triode
        else:
            return 2  # saturation


def get_nmos_region(vgs, vds, vth):
    if vgs < vth:
        return 0  # cut-off
    else:
        if vds < vgs - vth:
            return 1  # triode
        else:
            return 2  # saturation


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
        pp_class = getattr(pp_module, tb_dict["tb_class"])
        self.post_process = getattr(pp_class, tb_dict["post_process_function"])
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
        logger.debug(f"Creating a new design with {state=} ")
        output = self.template.render(**state)

        design_folder = os.path.join(self.gen_dir, new_fname)
        os.makedirs(design_folder, exist_ok=True)
        fpath = os.path.join(design_folder, new_fname + ".scs")
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

    def _parse_result(self, design_folder):

        logger.debug(f"Parsing results from design folder: {design_folder}")

        vals = get_values(os.path.join(design_folder, "output.log"))
        names = get_variable_names(os.path.join(design_folder, "output.log"))

        # print("length of names:", len(names))
        # Output: ['v(voutp)', 'i(v0)', '@nm0[gm]', '@nm0[ids]', '@nm0[vth]', 'vgs_nm0', 'vds_nm0',
        # '@nm1[gm]', '@nm1[ids]', '@nm1[vth]', 'vgs_nm1', 'vds_nm1',
        # '@nm2[gm]', '@nm2[ids]', '@nm2[vth]', 'vgs_nm2', 'vds_nm2',
        # '@nm3[gm]', '@nm3[ids]', '@nm3[vth]', 'vgs_nm3', 'vds_nm3',
        # '@nm4[gm]', '@nm4[ids]', '@nm4[vth]', 'v(vgs_nm4)', 'v(vds_nm4)',
        # '@nm5[gm]', '@nm5[ids]', '@nm5[vth]', 'vgs_nm5', 'vds_nm5',
        # '@nm6[gm]', '@nm6[ids]', '@nm6[vth]', 'vgs_nm6', 'vds_nm6',
        # '@nm7[gm]', '@nm7[ids]', '@nm7[vth]', 'vgs_nm7', 'vds_nm7',
        # '@nm8[gm]', '@nm8[ids]', '@nm8[vth]', 'vgs_nm8', 'vds_nm8',
        # '@nm9[gm]', '@nm9[ids]', '@nm9[vth]', 'v(vgs_nm9)', 'v(vds_nm9)',
        # '@nm10[gm]', '@nm10[ids]', '@nm10[vth]', 'v(vgs_nm10)', 'v(vds_nm10)']

        # Ensure the number of values matches the number of column names
        assert len(vals) == len(
            names
        ), "Mismatch between values and column names length"

        # turn "vals" and "names" into a pandas DataFrame
        df = pd.DataFrame([vals], columns=names)

        # Add region columns for PMOS transistors: nm5, nm6, nm2, nm1
        pmos_list = ["nm5", "nm6", "nm2", "nm1"]
        for pmos in pmos_list:
            # Get column names for Vgs, Vds, Vth
            vgs_col = f"vgs_{pmos}"
            vds_col = f"vds_{pmos}"
            vth_col = f"@{pmos}[vth]"
            # Some Vgs/Vds columns are named as v(vgs_nmX), v(vds_nmX)
            if vgs_col not in df.columns:
                vgs_col = f"v(vgs_{pmos})"
            if vds_col not in df.columns:
                vds_col = f"v(vds_{pmos})"
            # Compute region
            df[f"@{pmos}[region]"] = df.apply(
                lambda row: get_pmos_region(row[vgs_col], row[vds_col], -row[vth_col]),
                axis=1,
            )

        # Add region columns for NMOS transistors: nm8, nm7, nm3, nm0, nm10, nm9, nm4
        nmos_list = ["nm8", "nm7", "nm3", "nm0", "nm10", "nm9", "nm4"]
        for nmos in nmos_list:
            vgs_col = f"vgs_{nmos}"
            vds_col = f"vds_{nmos}"
            vth_col = f"@{nmos}[vth]"
            # Some Vgs/Vds columns are named as v(vgs_nmX), v(vds_nmX)
            if vgs_col not in df.columns:
                vgs_col = f"v(vgs_{nmos})"
            if vds_col not in df.columns:
                vds_col = f"v(vds_{nmos})"
            df[f"@{nmos}[region]"] = df.apply(
                lambda row: get_nmos_region(row[vgs_col], row[vds_col], row[vth_col]),
                axis=1,
            )

        # ===========================================================
        # Write a new file with the same netlist definition without .control block
        # this is needed for the ac simulation to work
        # we assume that the netlist file is in the design_folder and it has a .scs extension
        # ===========================================================
        if not os.path.exists(design_folder):
            raise FileNotFoundError(f"Design folder {design_folder} does not exist")

        # Find the netlist file in the design folder
        # Assuming there is only one .scs file in the design folder
        netlist_path = glob.glob(os.path.join(design_folder, "*.scs"))[0]
        new_netlist_path = os.path.join(design_folder, "netlist.scs")

        with open(netlist_path, "r") as fr, open(new_netlist_path, "w") as fw:
            for line in fr:
                # Remove the first line if it contains 'ngspice'
                if not line.startswith(".control"):
                    fw.write(line)
                else:
                    fw.write(".end")
                    break

        # Perform simulation & obtain performance metrics
        netlist = open(os.path.join(design_folder, "netlist.scs")).read()
        input_nodes = ["Vinn", "Vinp"]
        output_nodes = ["Voutn", "Voutp"]
        ac_netlist = setup_ac_simulation(netlist, input_nodes, output_nodes)
        run_ngspice_simulation(ac_netlist)
        bandwidth = compute_bandwidth("output_ac.dat")
        ubw, valid = compute_unity_gain_bandwidth("output_ac.dat")
        phase_margin = compute_phase_margin("output_ac.dat")
        ac_gain = compute_ac_gain("output_ac.dat")

        op_netlist = setup_op_simulation(netlist, input_nodes, measure_vars=["i(v0)"])
        run_ngspice_simulation(op_netlist)
        power = compute_power("output_op.dat")

        metrics = {}
        metrics["bandwidth"] = bandwidth
        metrics["ugbw"] = ubw
        metrics["pm"] = phase_margin
        metrics["ac_gain"] = ac_gain
        metrics["power"] = power
        metrics["valid"] = valid
        res = {"df": df, "metrics": metrics}
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
    CIR_YAML = "/home/pham/code/analog-ml/LEDRO/optimizer/working_current/spectre_simulator/spectre/specs_list_read/fully_differential_folded_cascode.yaml"
    with open(CIR_YAML, "r") as f:
        yaml_data = yaml.load(f, OrderedDictYAMLLoader)

    opamp_env = SpectreWrapper(tb_dict=yaml_data["measurement"]["testbenches"]["ac_dc"])

    data = "\nInitial Data:\nI got these points and their reward: metrics- gain:  5.77 , UGBW:  2.91e+04 , PM:  69.2 , power:  4.32e-12 with nA1: 7.45e-08, nB1: 6, nA2: 1.4e-07, nB2: 2, nA3: 3.75e-08, nB3: 3, nA4: 3.04e-07, nB4: 3, nA5: 3.72e-08, nB5: 4, nA6: 1.24e-07, nB6: 2, vbiasp1: 0.659, vbiasp2: 0.408, vbiasn0: 0.0525, vbiasn1: 0.016, vbiasn2: 0.352, vcm: 0.4, vdd: 0.8, tempc: 27 and transistor regions MM0 is in cut-off, MM1 is in cut-off, MM2 is in cut-off, MM3 is in cut-off, MM4 is in cut-off, MM5 is in cut-off, MM6 is in cut-off, MM7 is in cut-off, MM8 is in cut-off, MM9 is in cut-off, MM10 is in cut-off and reward -3.99. "
    state = OrderedDict({
        "nA1": 7.45e-08,
        "nB1": 6,
        "nA2": 1.4e-07,
        "nB2": 2,
        "nA3": 3.75e-08,
        "nB3": 3,
        "nA4": 3.04e-07,
        "nB4": 3,
        "nA5": 3.72e-08,
        "nB5": 4,
        "nA6": 1.24e-07,
        "nB6": 2,
        "vbiasp1": 0.659,
        "vbiasp2": 0.408,
        "vbiasn0": 0.0525,
        "vbiasn1": 0.016,
        "vbiasn2": 0.352,
        "vcm": 0.4,
        "vdd": 0.8,
        "tempc": 27,
    })
    state, specs, info = opamp_env._create_design_and_simulate(state)
    logger.info(state)
    logger.info(specs)

    from sample.random_sample_turbo_1 import Levy
    nA1_range = [10e-9, 900e-9]
    nB1_range = [1, 7]
    nA2_range = [10e-9, 900e-9]
    nB2_range = [1, 7]
    nA3_range = [10e-9, 900e-9]
    nB3_range = [1, 7]
    nA4_range = [10e-9, 900e-9]
    nB4_range = [1, 7]
    nA5_range = [10e-9, 900e-9]
    nB5_range = [1, 7]
    nA6_range = [10e-9, 900e-9]
    nB6_range = [1, 7]
    vbiasp1_range = [0.01, 0.8]
    vbiasp2_range = [0.01, 0.8]
    vbiasn0_range = [0.01, 0.8]
    vbiasn1_range = [0.01, 0.8]
    vbiasn2_range = [0.01, 0.8]
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
    vcm = 0.40
    vdd = 0.8
    tempc = 27
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
    f = Levy(17, params_id, specs_id, specs_ideal, vcm, vdd, tempc, ub, lb)

    # print (f())
    x = list(state.values())[:-3]
    x = np.array(x).reshape(-1)
    print (f.evaluate(x))

if __name__ == "__main__":
    main()
