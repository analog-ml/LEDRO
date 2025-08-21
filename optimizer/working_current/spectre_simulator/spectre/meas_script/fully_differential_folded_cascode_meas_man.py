from spectre_simulator.spectre.core import EvaluationEngine
import numpy as np
import pdb
import IPython
import scipy.interpolate as interp
import scipy.optimize as sciopt
import matplotlib.pyplot as plt
import globalsy


class OpampMeasMan(EvaluationEngine):

    def __init__(self, yaml_fname):
        EvaluationEngine.__init__(self, yaml_fname)

    def get_specs(self, results_dict, params):
        specs_dict = dict()
        ac_dc = results_dict["ac_dc"]
        for _, res, _ in ac_dc:
            specs_dict = res
        return specs_dict

    def compute_penalty(self, spec_nums, spec_kwrd):
        if type(spec_nums) is not list:
            spec_nums = [spec_nums]
        penalties = []
        for spec_num in spec_nums:
            penalty = 0
            spec_min, spec_max, w = self.spec_range[spec_kwrd]
            if spec_max is not None:
                if spec_num > spec_max:
                    penalty += w * abs(spec_num - spec_max) / abs(spec_num)
            if spec_min is not None:
                if spec_num < spec_min:
                    penalty += w * abs(spec_num - spec_min) / abs(spec_min)
            penalties.append(penalty)
        return penalties


class ACTB(object):

    @classmethod
    def process_ac(cls, results, params):
        dc_results = results["df"]

        ids_MM = []
        ids_MM.append(float("%.3g" % np.abs(dc_results["@nm0[ids]"].iloc[0])))
        ids_MM.append(float("%.3g" % np.abs(dc_results["@nm1[ids]"].iloc[0])))
        ids_MM.append(float("%.3g" % np.abs(dc_results["@nm2[ids]"].iloc[0])))
        ids_MM.append(float("%.3g" % np.abs(dc_results["@nm3[ids]"].iloc[0])))
        ids_MM.append(float("%.3g" % np.abs(dc_results["@nm4[ids]"].iloc[0])))
        ids_MM.append(float("%.3g" % np.abs(dc_results["@nm5[ids]"].iloc[0])))
        ids_MM.append(float("%.3g" % np.abs(dc_results["@nm6[ids]"].iloc[0])))
        ids_MM.append(float("%.3g" % np.abs(dc_results["@nm7[ids]"].iloc[0])))
        ids_MM.append(float("%.3g" % np.abs(dc_results["@nm8[ids]"].iloc[0])))
        ids_MM.append(float("%.3g" % np.abs(dc_results["@nm9[ids]"].iloc[0])))
        ids_MM.append(float("%.3g" % np.abs(dc_results["@nm10[ids]"].iloc[0])))

        gm_MM = []
        gm_MM.append(float("%.3g" % dc_results["@nm0[gm]"].iloc[0]))
        gm_MM.append(float("%.3g" % dc_results["@nm1[gm]"].iloc[0]))
        gm_MM.append(float("%.3g" % dc_results["@nm2[gm]"].iloc[0]))
        gm_MM.append(float("%.3g" % dc_results["@nm3[gm]"].iloc[0]))
        gm_MM.append(float("%.3g" % dc_results["@nm4[gm]"].iloc[0]))
        gm_MM.append(float("%.3g" % dc_results["@nm5[gm]"].iloc[0]))
        gm_MM.append(float("%.3g" % dc_results["@nm6[gm]"].iloc[0]))
        gm_MM.append(float("%.3g" % dc_results["@nm7[gm]"].iloc[0]))
        gm_MM.append(float("%.3g" % dc_results["@nm8[gm]"].iloc[0]))
        gm_MM.append(float("%.3g" % dc_results["@nm9[gm]"].iloc[0]))
        gm_MM.append(float("%.3g" % dc_results["@nm10[gm]"].iloc[0]))

        vgs_MM = []
        vgs_MM.append(round(dc_results["vgs_nm0"].iloc[0], 3))
        vgs_MM.append(round(dc_results["vgs_nm1"].iloc[0], 3))
        vgs_MM.append(round(dc_results["vgs_nm2"].iloc[0], 3))
        vgs_MM.append(round(dc_results["vgs_nm3"].iloc[0], 3))
        vgs_MM.append(round(dc_results["v(vgs_nm4)"].iloc[0], 3))
        vgs_MM.append(round(dc_results["vgs_nm5"].iloc[0], 3))
        vgs_MM.append(round(dc_results["vgs_nm6"].iloc[0], 3))
        vgs_MM.append(round(dc_results["vgs_nm7"].iloc[0], 3))
        vgs_MM.append(round(dc_results["vgs_nm8"].iloc[0], 3))
        vgs_MM.append(round(dc_results["v(vgs_nm9)"].iloc[0], 3))
        vgs_MM.append(round(dc_results["v(vgs_nm10)"].iloc[0], 3))

        vds_MM = []
        vds_MM.append(round(dc_results["vds_nm0"].iloc[0], 3))
        vds_MM.append(round(dc_results["vds_nm1"].iloc[0], 3))
        vds_MM.append(round(dc_results["vds_nm2"].iloc[0], 3))
        vds_MM.append(round(dc_results["vds_nm3"].iloc[0], 3))
        vds_MM.append(round(dc_results["v(vds_nm4)"].iloc[0], 3))
        vds_MM.append(round(dc_results["vds_nm5"].iloc[0], 3))
        vds_MM.append(round(dc_results["vds_nm6"].iloc[0], 3))
        vds_MM.append(round(dc_results["vds_nm7"].iloc[0], 3))
        vds_MM.append(round(dc_results["vds_nm8"].iloc[0], 3))
        vds_MM.append(round(dc_results["v(vds_nm9)"].iloc[0], 3))
        vds_MM.append(round(dc_results["v(vds_nm10)"].iloc[0], 3))

        region_of_operation_MM = []
        region_of_operation_MM.append(int(dc_results["@nm0[region]"].iloc[0]))
        region_of_operation_MM.append(int(dc_results["@nm1[region]"].iloc[0]))
        region_of_operation_MM.append(int(dc_results["@nm2[region]"].iloc[0]))
        region_of_operation_MM.append(int(dc_results["@nm3[region]"].iloc[0]))
        region_of_operation_MM.append(int(dc_results["@nm4[region]"].iloc[0]))
        region_of_operation_MM.append(int(dc_results["@nm5[region]"].iloc[0]))
        region_of_operation_MM.append(int(dc_results["@nm6[region]"].iloc[0]))
        region_of_operation_MM.append(int(dc_results["@nm7[region]"].iloc[0]))
        region_of_operation_MM.append(int(dc_results["@nm8[region]"].iloc[0]))
        region_of_operation_MM.append(int(dc_results["@nm9[region]"].iloc[0]))
        region_of_operation_MM.append(int(dc_results["@nm10[region]"].iloc[0]))

        metrics = results["metrics"]
        gain = metrics["ac_gain"]
        ugbw, valid = metrics["ugbw"], metrics["valid"]
        phm = metrics["pm"]
        power = metrics["power"]

        # this condition will be used!
        if globalsy.counterrrr < 200:
            f = open("out1.txt", "a")
            print(
                "metrics-",
                "gain: ",
                f"{float(gain):.3}",
                ", UGBW: ",
                f"{float(ugbw):.3}",
                ", PM: ",
                f"{float(phm):.3}",
                ", power: ",
                f"{float(power):.3}",
                valid,
                file=f,
            )
            f.close()
        elif globalsy.counterrrr < 1200:
            f = open("/path/to/optimizer/out11.txt", "a")
            print(
                "metrics-",
                "gain: ",
                f"{float(gain):.3}",
                ", UGBW: ",
                f"{float(ugbw):.3}",
                ", PM: ",
                f"{float(phm):.3}",
                ", power: ",
                f"{float(power):.3}",
                valid,
                file=f,
            )
            f.close()
        elif globalsy.counterrrr < 2000:
            f = open("/path/to/optimizer/out12.txt", "a")
            print(
                "metrics-",
                "gain: ",
                f"{float(gain):.3}",
                ", UGBW: ",
                f"{float(ugbw):.3}",
                ", PM: ",
                f"{float(phm):.3}",
                ", power: ",
                f"{float(power):.3}",
                valid,
                file=f,
            )
            f.close()
        results = dict(
            gain=gain,
            funity=ugbw,
            pm=phm,
            power=power,
            valid=valid,
            zregion_of_operation_MM=region_of_operation_MM,
            zzgm_MM=gm_MM,
            zzids_MM=ids_MM,
            zzvds_MM=vds_MM,
            zzvgs_MM=vgs_MM,
        )
        f.close()
        return results

    @classmethod
    def find_dc_gain(self, vout):
        return np.abs(vout)[0]

    @classmethod
    def find_ugbw(self, freq, vout):
        gain = np.abs(vout)
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        if valid:
            return ugbw, valid
        else:
            return freq[0], valid

    @classmethod
    def find_phm(self, freq, vout):
        gain = np.abs(vout)
        phase = np.angle(vout, deg=False)
        phase = np.unwrap(phase)  # unwrap the discontinuity
        phase = np.rad2deg(phase)  # convert to degrees
        #
        # plt.subplot(211)
        # plt.plot(np.log10(freq[:200]), 20*np.log10(gain[:200]))
        # plt.subplot(212)
        # plt.plot(np.log10(freq[:200]), phase)
        # plt.show()

        phase_fun = interp.interp1d(freq, phase, kind="quadratic")
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        if valid:
            if phase[0] > 150:
                return phase_fun(ugbw)
            else:
                return 180 + phase_fun(ugbw)
            # if phase_fun(ugbw) > 0:
            #    return -180+phase_fun(ugbw)
            # else:
            #    return 180 + phase_fun(ugbw)
        else:
            return -180

    @classmethod
    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop), True
        except ValueError:
            # avoid no solution
            # if abs(fzero(xstart)) < abs(fzero(xstop)):
            #     return xstart
            return xstop, False


if __name__ == "__main__":

    yname = "/path/to/optimizer/working_current/spectre_simulator/spectre/specs_list_read/fully_differential_folded_cascode.yaml"
    eval_core = OpampMeasMan(yname)

    designs = eval_core.generate_data_set(n=1)
