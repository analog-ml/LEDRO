import re


def convert_scs_to_spice(scs_lines):
    spice_lines = []

    param_block = []
    for line in scs_lines:
        # Skip simulator, saveOptions, or Spectre-specific directives
        if line.strip().startswith(
            (
                "simulator",
                "simulatorOptions",
                "saveOptions",
                "dcOp",
                "dcOpInfo",
                "modelParameter",
                "element",
                "outputParameter",
                "designParamVals",
                "primitives",
                "subckts",
                "save",
                "//",
                "*.",
            )
        ):
            continue

        # Convert 'parameters' to .param
        if line.strip().startswith("parameters"):
            line = re.sub(r"parameters", ".param", line)
            param_block.append(line)
            continue
        if param_block:
            if line.strip().startswith("include") or line.strip() == "":
                spice_lines += param_block
                param_block = []
            else:
                param_block.append(line)
                continue

        # Convert 'include' to SPICE-style
        if "include" in line:
            line = line.replace("include", ".include")

        # Convert vsource lines to SPICE-style
        line = re.sub(
            r"(\w+)\s+\(([^)]+)\)\s+vsource\s+dc=([\w\.]+)\s+type=dc",
            r"\1 \2 DC \3",
            line,
        )
        line = re.sub(
            r"(\w+)\s+\(([^)]+)\)\s+vsource\s+type=dc\s+dc=([\w\.]+)",
            r"\1 \2 DC \3",
            line,
        )
        line = re.sub(
            r"(\w+)\s+\(([^)]+)\)\s+vsource\s+type=dc\s+mag=([\w\.]+)",
            r"\1 \2 DC \3",
            line,
        )

        # Convert vcvs
        line = re.sub(
            r"(\w+)\s+\(([^)]+)\)\s+vcvs\s+gain=([\-\.\w]+)", r"\1 \2 E \3", line
        )

        # Convert MM MOSFET instances
        line = re.sub(
            r"^MM(\w+)\s+([\w\s!]+)\s+(\w+)\s+l=([\w\.]+)\s+nfin=([\w\.]+)",
            r"M\1 \2 \3 W=1 L=\4 nf=\5",
            line,
        )

        # Replace Spectre-specific node names like vdd! or gnd!
        line = line.replace("vdd!", "vdd")
        line = line.replace("gnd!", "0")  # SPICE uses 0 as ground

        spice_lines.append(line.strip())

    # Append end directive if not already there
    if not any(".end" in line.lower() for line in spice_lines):
        spice_lines.append(".end")

    return spice_lines


def main():
    input_path = "input.scs"
    input_path = "/tmp/ledro/designs_fully_differential_folded_cascode2482/fully_differential_folded_cascode2482_4.25_4.0_2.85_4.0_2.8_4.0_5.69_3.0_4.94_4.0_4.74_4.0_0.41_0.32_0.48_0.23_0.58_0.4_0.8_27.0/fully_differential_folded_cascode2482_4.25_4.0_2.85_4.0_2.8_4.0_5.69_3.0_4.94_4.0_4.74_4.0_0.41_0.32_0.48_0.23_0.58_0.4_0.8_27.0.scs"
    output_path = "output.cir"

    with open(input_path, "r") as f:
        scs_lines = f.readlines()

    spice_lines = convert_scs_to_spice(scs_lines)

    with open(output_path, "w") as f:
        f.write("* Converted from Spectre .scs to SPICE .cir\n")
        for line in spice_lines:
            f.write(line + "\n")

    print(f"Converted netlist saved to: {output_path}")


if __name__ == "__main__":
    main()
