# LEDRO
This repository is the code for [LEDRO: LLM-Enhanced Design Space Reduction and Optimization for Analog Circuits](https://arxiv.org/abs/2411.12930). 

Traditional approaches for designing analog circuits are time-consuming and require significant human expertise. Existing automation efforts using methods like Bayesian Optimization (BO) and Reinforcement Learning (RL) are sub-optimal and costly to generalize across different topologies and technology nodes. In our work, we introduce a novel approach, LEDRO, utilizing Large Language Models (LLMs) in conjunction with optimization techniques to iteratively refine the design space for analog circuit sizing. LEDRO is highly generalizable compared to other RL and BO baselines, eliminating the need for design annotation or model training for different topologies or technology nodes. We conduct a comprehensive evaluation of our proposed framework and baseline on 22 different Op-Amp topologies across four FinFET technology nodes. Results demonstrate the superior performance of LEDRO as it outperforms our best baseline by an average of 13% FoM improvement with 2.15x speed-up on low complexity Op-Amps and 48% FoM improvement with 1.7x speed-up on high complexity Op-Amps. This highlights LEDRO's effective performance, efficiency, and generalizability.

The top level directory contains two sub-directories:

optimizer/: contains all of the optimizer code, netlist, simulator interface, and result extraction code\
LLM/: contains all the LLM code with template prompts

To run the code, please run the main.sh script from the optimizer folder.

Note that the results shown in the paper include those from:
1) Cadence Spectre simulator. Spectre requires a license.
2) LLaMa-3-70B using IBM BAM, but can be modified to work with other hosted LLMs.

This repository also includes all files generated for the Section II-D: Case Study: End-to-end example of 7nm differential folded cascode amplifier, and Fig. 5 in the paper.\
The netlist templates include all the netlist benchmarks in each technology node mentioned in the paper (Fig. 7, Fig. 8). 

We would like to thank:
1) [AutoCkt](https://github.com/ksettaluri6/AutoCkt/) for the RL baseline code, and simulator interfacing script.
2) [GANA_circuit_data](https://github.com/kkunal1408/GANA_circuit_data) for baseline netlists we could modify for our task.
3) [TuRBO](https://github.com/uber-research/TuRBO/tree/master) for our baseline optimizer code.


If our work is useful to you, please cite our [paper](https://arxiv.org/abs/2411.12930):

```
@article{kochar2024ledro,
  title={LEDRO: LLM-Enhanced Design Space Reduction and Optimization for Analog Circuits},
  author={Kochar, Dimple Vijay and Wang, Hanrui and Chandrakasan, Anantha and Zhang, Xin},
  journal={arXiv preprint arXiv:2411.12930},
  year={2024}
}
  ```


