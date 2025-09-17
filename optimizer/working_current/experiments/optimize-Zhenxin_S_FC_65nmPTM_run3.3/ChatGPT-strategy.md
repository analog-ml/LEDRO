You: 

You are an experienced analog circuit designer. You are given a SPICE netlist together with hierarchical building block information (transistor names and their functions in the given netlist). You need to do sizing optimization with the following target: Gain >= 1400, UGBW >= 1.0e8, Phrase Margin >= 70, Power <= 1mA with the 65nm PTM technology node. As simultaneously optimize all transistors and design targets with conventional methods (i.e., reinforcement learning or Bayesian Optimization) will result in many redundant simulation steps. Your task is to derive a efficient sizing optimization strategy, based on selecting which block to focus on to each steps, and the weight coefficients to construct Figure of Merits (for example, FoM = w\_g \* (current gain - target gain) + w\_ugbw ( current ugbw - target ugbw) + w\_pm (current phrase margin - target phrase margin) + w\_power (current power - target power)). The generated strategy should be target to quickly achieve the design targets with minimal simulation steps required. Try to output sequence of weight coefficients that avoid suddenly discrupted transitions (quickly jumps or drops) the overall FoM during training. Normalize weight coefficients between 0 and 1, and sum of weight coefficients is 1 Please return your strategy the following format \[ { "selected\_block": \[1, 2\], # Hierarchal Level 1, 2nd subcircuit block, "weight\_coeffients": \[3, 1, 0, 0\], # weight coefficients for gain, ugbw, phrase margin, and power, respectively. "purpose": (explain the purpose / objective here) "stop\_condition": \["gain", "larger", 60\] # add stopping condition here, after achiving this condition we will move the next phase }, # phase 1 { "selected\_block": \[2, 1\], # Hierarchal Level 2, 1st subcircuit block, "weight\_coeffients": \[0, 0, 0, 1\], # weight coefficients for gain, ugbw, phrase margin, and power, respectively. "purpose": (explain the purpose / objective here) "stop\_condition": \["ugbw", "larger", 1.2e6\] # add stopping condition here, after achiving this condition we will move the next phase }, # phase 2 .... \] Input SPICE netlist Parameters .param tempc=25.0 .param wm12={{w\_m12}}n lm12=130n mm12=1 .param wm3={{w\_m3}}n lm3=130n mm3=1 .param wm45={{w\_m45}}n lm45=130n mm45=1 .param wm67={{w\_m67}}n lm67=130n mm67=1 .param wm89={{w\_m89}}n lm89=130n mm89=1 .param wm1011={{w\_m1011}}n lm1011=130n mm1011=1 .param vbp1={{vbp1}} .param vbp2={{vbp2}} .param vbn1={{vbn1}} .param vbn2={{vbn2}} .param vdd=1.2 .param vcm=0.6 M3 N004 Vbp1 VDD VDD pmos W={wm3} L={lm3} m={mm3} M4 N002 N001 VDD VDD pmos W={wm45} L={lm45} m={mm45} M5 N003 N001 VDD VDD pmos W={wm45} L={lm45} m={mm45} M7 Vout Vbp2 N003 N003 pmos W={wm67} L={lm67} m={mm67} M6 N001 Vbp2 N002 N002 pmos W={wm67} L={lm67} m={mm67} M1 N006 Vinp N004 N004 pmos W={wm12} L={lm12} m={mm12} M2 N005 Vinn N004 N004 pmos W={wm12} L={lm12} m={mm12} M8 N001 Vbn1 N006 N006 nmos W={wm89} L={lm89} m={mm89} M9 Vout Vbn1 N005 N005 nmos W={wm89} L={lm89} m={mm89} M10 N006 Vbn2 0 0 nmos W={wm1011} L={lm1011} m={mm1011} M11 N005 Vbn2 0 0 nmos W={wm1011} L={lm1011} m={mm1011} vin in 0 dc=0 ac=1.0 ein1 Vinp cm in 0 0.5 ein2 Vinn cm in 0 -0.5 vcm cm 0 dc={vcm} vdd VDD 0 dc=1.2 vss 0 VSS dc=0 Ccomp N001 Vout {{cc}}p Cload Vout 0 1p VBP1 Vbp1 0 DC {vbp1} VBP2 Vbp2 0 DC {vbp2} VBN1 Vbn1 0 DC {vbn1} VBN2 Vbn2 0 DC {vbn2} Hierarchical Building Block Information # Hierarchial 1 — Top logical blocks - \*\*Subcircuit 1: Bottom sinks / tail currents\*\* - Devices: \*\*M10, M11\*\* - \*\*Subcircuit 2: Input tail / bias PMOS\*\* - Devices: \*\*M3\*\* - \*\*Subcircuit 3: Differential input pair\*\* - Devices: \*\*M1, M2\*\* - \*\*Subcircuit 4: PMOS cascode / top load network\*\* - Devices: \*\*M4, M5, M6, M7\*\* - \*\*Subcircuit 5: NMOS folded-path / single-ended converter\*\* - Devices: \*\*M8, M9\*\* - \*\*Subcircuit 6: Compensation & load\*\* - Devices/Parts: \*\*Ccomp (between N001 and Vout), Cload (Vout to 0)\*\* - \*\*Subcircuit 7: Supplies / bias nets (explicit sources)\*\* - Parts: \*\*bias nodes VBP1, VBP2, VBN1, VBN2 (nets implicitly driven)\*\* --- # Hierarchial 2 — Functional stages - \*\*Stage A — Bias & current reference stage\*\* - Devices: \*\*M10, M11, M3\*\* (and bias nets VBN2, VBP1) - \*\*Stage B — Input differential stage\*\* - Devices: \*\*M1, M2\*\* (connected to M3 tail and node N006/N005) - \*\*Stage C — Folded-cascode gain stage\*\* - Devices: \*\*M4, M5, M6, M7, M8, M9\*\* (PMOS cascodes + NMOS folded-path) - \*\*Stage D — Output & compensation interaction\*\* - Devices/parts: \*\*M7, M9, Ccomp, Cload, Vout net\*\* - \*\*Stage E — Global bias trimming and matching\*\* - Devices: all bias nets --- # Hierarchial 3 — Fine device groups - \*\*Group 1 — Input matched pair\*\* - Devices: \*\*M1, M2\*\* - \*\*Group 2 — Top PMOS mirror / cascode symmetry\*\* - Devices: \*\*M4, M5\*\* , \*\*M6,M7\*\* - \*\*Group 3 — Folded-path NMOS pair and bias\*\* - Devices: \*\*M8, M9\*\* and bias net \*\*VBN1\*\* - \*\*Group 4 — Bottom sink matching & mirror ratio control\*\* - Devices: \*\*M10, M11\*\* and bias net \*\*VBN2\*\* - \*\*Group 5 — Compensation and output tuning\*\* - Parts: \*\*Ccomp (N001–Vout), Cload\*\* and devices \*\*M7, M9\*\* Let's think step by step.

ChatGPT: 

Thought for 1m 10s

\[  
{  
"selected\_block": \[2, 1\],  
"weight\_coeffients": \[0.15, 0.15, 0.10, 0.60\],  
"purpose": "Establish bias and tail currents (M10, M11, M3). Bring steady-state power consumption inside the 1mA budget while keeping bias headroom for later gm increases. This phase prevents later large power-induced FoM jumps.",  
"stop\_condition": \["power", "smaller", 0.98e-3\]  
},  
{  
"selected\_block": \[2, 2\],  
"weight\_coeffients": \[0.20, 0.25, 0.10, 0.45\],  
"purpose": "Size the input differential pair (M1, M2): increase useful gm for UGBW improvement while monitoring the impact on power and input capacitance. Keep power weight non-negligible to avoid overshooting current.",  
"stop\_condition": \["ugbw", "larger", 6.0e7\]  
},  
{  
"selected\_block": \[2, 3\],  
"weight\_coeffients": \[0.30, 0.30, 0.05, 0.35\],  
"purpose": "Tune folded-cascode gain stage (M4, M5, M6, M7, M8, M9): increase output resistance (gain) via cascode sizing and mirror ratios while keeping a reasonable UGBW. Maintain a measured power penalty allowance to avoid violating budget.",  
"stop\_condition": \["gain", "larger", 700\]  
},  
{  
"selected\_block": \[2, 4\],  
"weight\_coeffients": \[0.35, 0.35, 0.05, 0.25\],  
"purpose": "Adjust compensation / output interaction (Ccomp, M7, M9, Cload): move poles/zeros to recover UGBW and shape phase response. Objective is to raise UGBW and avoid phase margin collapse as gain grows.",  
"stop\_condition": \["phrase margin", "larger", 60\]  
},  
{  
"selected\_block": \[3, 2\],  
"weight\_coeffients": \[0.35, 0.40, 0.10, 0.15\],  
"purpose": "Fine tune top PMOS mirror / cascode symmetry (Group 2: M4,M5,M6,M7). Focus shifts towards UGBW and gain jointly: trim cascode bias, W/L ratios and mirror ratios to push toward the final gain target while limiting extra pole introduction.",  
"stop\_condition": \["gain", "larger", 1200\]  
},  
{  
"selected\_block": \[2, 5\],  
"weight\_coeffients": \[0.40, 0.35, 0.15, 0.10\],  
"purpose": "Global bias trimming & final matching (Stage E): small coordinated adjustments across bias nets and mirrored ratios to converge all metrics together (gain, UGBW, phase margin) while keeping power within the budget. This is the final convergence stage.",  
"stop\_condition": \["gain", "larger", 1400\]  
}  
\]