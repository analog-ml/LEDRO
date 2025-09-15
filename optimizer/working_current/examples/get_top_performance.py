with open("performance+deviceparams.log", "r") as f:
    lines = f.readlines()
    print(f"Total lines in performance+deviceparams.log: {len(lines)}")
    # only select the first 200 lines

    selected_lines = []
    for line in lines[:200]:
        # sample
        # reward: -4, nA1: 1.06e-08, nB1: 3, nA2: 2.56e-08, nB2: 3, nA3: 1.78e-08, nB3: 3, nA4: 2.57e-08, nB4: 3, nA5: 3.79e-08, nB5: 5, nA6: 3.2e-08, nB6: 4, vbiasp1: 0.521, vbiasp2: 0.407, vbiasn0: 0.581, vbiasn1: 0.224, vbiasn2: 0.531, vcm: 0.4, vdd: 0.8, tempc: 27, MM0 is in cut-off, MM1 is in triode, MM2 is in triode, MM3 is in cut-off, MM4 is in triode, MM5 is in triode, MM6 is in triode, MM7 is in cut-off, MM8 is in cut-off, MM9 is in cut-off, MM10 is in cut-off
        # get the reward value
        reward = line.split(",")[0].split(":")[1].strip()
        selected_lines.append((reward, line.strip()))

    # get top 10 rewards
    top_rewards = sorted(selected_lines, key=lambda x: -float(x[0]))[:5]

    # write to file
    for reward, line in top_rewards:
        # print(f"Reward: {reward}, Line: {line}")
        print(line)
