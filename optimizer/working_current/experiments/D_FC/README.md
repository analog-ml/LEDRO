I run the pure TURBO1 with the "reasonable" parameters range

```python
nA1_range = (1.0e-8, 4.0e-8)
nB1_range = (3, 5)
nA2_range = (1.0e-8, 4.0e-8)
nB2_range = (2, 4)
nA3_range = (1.0e-8, 4.0e-8)
nB3_range = (3, 5)
nA4_range = (1.0e-8, 4.0e-8)
nB4_range = (2, 4)
nA5_range = (1.0e-8, 4.0e-8)
nB5_range = (4, 6)
nA6_range = (1.0e-8, 4.0e-8)
nB6_range = (3, 5)
nA7_range = (1.0e-8, 4.0e-8)
nB7_range = (1, 7)
nA8_range = (1.0e-8, 4.0e-8)
nB8_range = (1, 7)
nA9_range = (1.0e-8, 4.0e-8)
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
```

```python
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
```

- The result is saved at `D_FC/data_15`



```
*fully_differential_folded_cascode.css 

include "PTM-MG/modelfiles/lstp/7nfet.pm" 
include "PTM-MG/modelfiles/lstp/7pfet.pm"

* Parameters
.param tempc={{tempc}}
.param nA1={{nA1}} nB1={{nB1}}
.param nA2={{nA2}} nB2={{nB2}}
.param nA3={{nA3}} nB3={{nB3}}
.param nA4={{nA4}} nB4={{nB4}}
.param nA5={{nA5}} nB5={{nB5}}
.param nA6={{nA6}} nB6={{nB6}}
.param vdd={{vdd}} vcm={{vcm}} vbiasp1={{vbiasp1}} vbiasp2={{vbiasp2}}
.param vbiasn0={{vbiasn0}} vbiasn1={{vbiasn1}} vbiasn2={{vbiasn2}}

NM6 Voutp Vbiasp2 net23 vdd pfet L={nA1} NFIN={nB1}
NM5 Voutn Vbiasp2 net24 vdd pfet L={nA1} NFIN={nB1}
NM2 net23 Vbiasp1 vdd vdd pfet L={nA2} NFIN={nB2}
NM1 net24 Vbiasp1 vdd vdd pfet L={nA2} NFIN={nB2}
NM8 Voutp Vbiasn2 net27 0 nfet L={nA3} NFIN={nB3}
NM7 Voutn Vbiasn2 net25 0 nfet L={nA3} NFIN={nB3}
NM3 net24 Vinp net13 0 nfet L={nA4} NFIN={nB4}
NM0 net23 Vinn net13 0 nfet L={nA4} NFIN={nB4}
NM10 net27 Vbiasn1 0 0 nfet L={nA5} NFIN={nB5}
NM9 net25 Vbiasn1 0 0 nfet L={nA5} NFIN={nB5}
NM4 net13 Vbiasn0 0 0 nfet L={nA6} NFIN={nB6}

* Voltage sources
* VS gnd 0 DC 0
V0 vdd 0 DC {vdd}
V2 in 0 DC 0 AC 1
E1 Vinp cm in 0 0.5
E0 Vinn cm in 0 -0.5
V1 cm 0 DC {vcm}
VP1 Vbiasp1 0 DC {vbiasp1}
VP2 Vbiasp2 0 DC {vbiasp2}
VN Vbiasn0 0 DC {vbiasn0}
VN1 Vbiasn1 0 DC {vbiasn1}
VN2 Vbiasn2 0 DC {vbiasn2}

.control
op
pre_osdi /home/pham/shared_files/ngspice/osdilibs/bsimcmg.osdi
set xbrushwidth=3
set filetype=ascii
run

let vgs_nm0 = v(Vinn) - v(net13)
let vds_nm0 = v(net23) - v(net13)

let vgs_nm1 = v(Vbiasp1) - v(vdd)
let vds_nm1 = v(net24) - v(vdd)

let vgs_nm2 = v(Vbiasp1) - v(vdd)
let vds_nm2 = v(net23) - v(vdd)

let vgs_nm3 = v(Vinp) - v(net13)
let vds_nm3 = v(net24) - v(net13)

let vgs_nm4 = v(Vbiasn0) 
let vds_nm4 = v(net13) 

let vgs_nm5 = v(Vbiasp2) - v(net24)
let vds_nm5 = v(Voutn) - v(net24)

let vgs_nm6 = v(Vbiasp2) - v(net23)
let vds_nm6 = v(Voutp) - v(net23)

let vgs_nm7 = v(Vbiasn2) - v(net25)
let vds_nm7 = v(Voutn) - v(net25)

let vgs_nm8 = v(Vbiasn2) - v(net27)
let vds_nm8 = v(Voutp) - v(net27)

let vgs_nm9 = v(Vbiasn1)
let vds_nm9 = v(net25)

let vgs_nm10 = v(Vbiasn1) 
let vds_nm10 = v(net27) 

write output.log I(V0) @nm0[gm] @nm0[ids] @nm0[vth] vgs_nm0 vds_nm0
+ @nm1[gm] @nm1[ids] @nm1[vth] vgs_nm1 vds_nm1
+ @nm2[gm] @nm2[ids] @nm2[vth] vgs_nm2 vds_nm2 
+ @nm3[gm] @nm3[ids] @nm3[vth] vgs_nm3 vds_nm3
+ @nm4[gm] @nm4[ids] @nm4[vth] vgs_nm4 vds_nm4
+ @nm5[gm] @nm5[ids] @nm5[vth] vgs_nm5 vds_nm5 
+ @nm6[gm] @nm6[ids] @nm6[vth] vgs_nm6 vds_nm6
+ @nm7[gm] @nm7[ids] @nm7[vth] vgs_nm7 vds_nm7
+ @nm8[gm] @nm8[ids] @nm8[vth] vgs_nm8 vds_nm8
+ @nm9[gm] @nm9[ids] @nm9[vth] vgs_nm9 vds_nm9
+ @nm10[gm] @nm10[ids] @nm10[vth] vgs_nm10 vds_nm10

quit
.endc

.end


This is a fully_differential_folded_cascode amplifier, using 7nm PTM finfets. Each transistor has a length parameter l and number of fins nfin. nAx and nBx are variable numbers for these. Vbias are biasing voltages. As we change them, our performance changes. We are considering the gain, phase margin (PM), unity gain bandwidth (UGBW) and power.
Try to get all transistors in what you think will be the correct region of operation by getting their role from the netlist provided earlier and changing parameters for sizing and biasing. First step is to analyze this netlist. Do you understand the role of each transistor in this netlist? 


Problem Description: \nThis is a fully_differential_folded_cascode amplifier, using 7nm PTM finfets. Each transistor has a length parameter l and number of fins nfin. nAx and nBx are variable numbers for these. Vbias are biasing voltages. As we change them, our performance changes. We are considering the gain, phase margin (PM), unity gain bandwidth (UGBW) and power.
Initial Data:
I got these points and their reward: 
reward: -2.95, nA1: 1.09e-08, nB1: 4, nA2: 2.49e-08, nB2: 4, nA3: 3.22e-08, nB3: 4, nA4: 2.41e-08, nB4: 2, nA5: 3.75e-08, nB5: 5, nA6: 1.96e-08, nB6: 3, vbiasp1: 0.407, vbiasp2: 0.4, vbiasn0: 0.478, vbiasn1: 0.351, vbiasn2: 0.625, vcm: 0.4, vdd: 0.8, tempc: 27, MM0 is in cut-off, MM1 is in triode, MM2 is in triode, MM3 is in cut-off, MM4 is in cut-off, MM5 is in triode, MM6 is in triode, MM7 is in cut-off, MM8 is in cut-off, MM9 is in cut-off, MM10 is in cut-off
reward: -2.95, nA1: 3.66e-08, nB1: 4, nA2: 1.8e-08, nB2: 3, nA3: 1.51e-08, nB3: 4, nA4: 1.93e-08, nB4: 3, nA5: 3.49e-08, nB5: 5, nA6: 3.4e-08, nB6: 4, vbiasp1: 0.398, vbiasp2: 0.297, vbiasn0: 0.525, vbiasn1: 0.261, vbiasn2: 0.452, vcm: 0.4, vdd: 0.8, tempc: 27, MM0 is in cut-off, MM1 is in triode, MM2 is in triode, MM3 is in cut-off, MM4 is in cut-off, MM5 is in triode, MM6 is in triode, MM7 is in cut-off, MM8 is in cut-off, MM9 is in cut-off, MM10 is in cut-off
reward: -2.95, nA1: 3.34e-08, nB1: 4, nA2: 3.32e-08, nB2: 4, nA3: 2.91e-08, nB3: 4, nA4: 2.82e-08, nB4: 4, nA5: 1.79e-08, nB5: 5, nA6: 1.45e-08, nB6: 4, vbiasp1: 0.397, vbiasp2: 0.355, vbiasn0: 0.436, vbiasn1: 0.253, vbiasn2: 0.523, vcm: 0.4, vdd: 0.8, tempc: 27, MM0 is in cut-off, MM1 is in triode, MM2 is in triode, MM3 is in cut-off, MM4 is in cut-off, MM5 is in triode, MM6 is in triode, MM7 is in cut-off, MM8 is in cut-off, MM9 is in cut-off, MM10 is in cut-off
reward: -2.96, nA1: 2.63e-08, nB1: 5, nA2: 2.1e-08, nB2: 2, nA3: 3.8e-08, nB3: 5, nA4: 1.73e-08, nB4: 3, nA5: 1.26e-08, nB5: 6, nA6: 2.49e-08, nB6: 3, vbiasp1: 0.378, vbiasp2: 0.274, vbiasn0: 0.458, vbiasn1: 0.291, vbiasn2: 0.548, vcm: 0.4, vdd: 0.8, tempc: 27, MM0 is in cut-off, MM1 is in triode, MM2 is in triode, MM3 is in cut-off, MM4 is in cut-off, MM5 is in triode, MM6 is in triode, MM7 is in cut-off, MM8 is in cut-off, MM9 is in cut-off, MM10 is in cut-off
reward: -2.96, nA1: 1.3e-08, nB1: 4, nA2: 2.86e-08, nB2: 2, nA3: 3.72e-08, nB3: 4, nA4: 3.39e-08, nB4: 2, nA5: 3.32e-08, nB5: 5, nA6: 1.38e-08, nB6: 5, vbiasp1: 0.377, vbiasp2: 0.28, vbiasn0: 0.425, vbiasn1: 0.391, vbiasn2: 0.56, vcm: 0.4, vdd: 0.8, tempc: 27, MM0 is in cut-off, MM1 is in triode, MM2 is in triode, MM3 is in cut-off, MM4 is in cut-off, MM5 is in triode, MM6 is in triode, MM7 is in cut-off, MM8 is in cut-off, MM9 is in cut-off, MM10 is in cut-off

Final Goal:
Our goal is to get reward = 0. Reward gets more negative when the gain achieved < target gain, UGBW achieved < target UGBW, PM achieved < target PM, power achieved > target power. Basically, you are not penalized for doing better (getting more than target gain, getting lower power, etc). Try to get all transistors in what you think will be the correct region of operation by getting their role from the netlist provided earlier and changing parameters for sizing and biasing. 
What you need to do and response format guidelines:\nWhat parameters should I change to obtain gain: 3300, funity: 20.0e+6, pm: 70, power: 10e-6? Give me ranges like nA = [#number, #number], etc - i.e. exact ranges for each parameter except vdd, vcm, temp. Note that nB are fins so they are integers, nA are lengths, and Vbias are voltages.
Give me exact ranges and not a strategy in the ranges as below:\n nA1:[10e-9, 400e-9],  nB1:[1, 7],  nA2:[10e-9, 400e-9],  nB2:[1, 7],  nA3:[10e-9, 400e-9],  nB3:[1, 7],  nA4:[10e-9, 400e-9],  nB4:[1, 7],  nA5:[10e-9, 400e-9],  nB5:[1, 7],  nA6:[10e-9, 400e-9],  nB6:[1, 7],  vbiasp1:[0.1, 0.8],  vbiasp2:[0.1, 0.8],  vbiasn0:[0.1, 0.8],  vbiasn1:[0.1, 0.8],  vbiasn2:[0.1, 0.8], 

Stick strictly to inside these ranges. Format your answer as \"parameter: [value, value]\", for example \"nA1: [5, 6]\", every parameter in a new line. That's the only output I want, without any explanations. Keep in mind the initial data you are provided, choose ranges based on that information as well as your knowledge of analog circuit design and amplifiers. Recognize the role of each transistor here (expecially those with same parameters which might form a symmetric pair), and which region of operation each transistor should operate in for this netlist. I will explore in the ranges you provide

```


Answer
```
nA1: [20e-9, 80e-9]
nB1: [3, 6]
nA2: [20e-9, 100e-9]
nB2: [2, 5]
nA3: [20e-9, 80e-9]
nB3: [3, 6]
nA4: [20e-9, 100e-9]
nB4: [2, 5]
nA5: [20e-9, 100e-9]
nB5: [3, 6]
nA6: [20e-9, 100e-9]
nB6: [2, 5]
vbiasp1: [0.35, 0.55]
vbiasp2: [0.35, 0.55]
vbiasn0: [0.35, 0.55]
vbiasn1: [0.25, 0.45]
vbiasn2: [0.45, 0.65]

```