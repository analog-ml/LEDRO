import numpy as np

current_specs = np.array([-1.73082230e+01  ,1.25892541e+00,  3.05105953e-01,  6.32388330e-12])
target_specs = np.array([3300, 20_000_000.0, 70, 10e-6]) # gain, funity, pm, power

def lookup( spec: list[float], goal_spec: list[float]) -> np.ndarray:
    """
    Normalize the specifications based on their ideal values.

    This function normalizes the specifications by calculating the relative difference between the current specification values and the ideal values.
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
    return norm_spec

reward = 0.0
norm_specs = lookup(current_specs, target_specs)
specs_id = ["gain", "funity", "pm", "ibias_max"]

for i, rel_spec in enumerate(norm_specs):
    # For power, smaller is better
    # For gain, larger (compared to the target/goal) is better
    # For other specs (pm, ugbw, etc.), smaller is better

    if specs_id[i] == "power" and rel_spec > 0:
        reward += np.abs(rel_spec) # /10

    elif specs_id[i] == "gain" and rel_spec < 0:
        reward += 3 * np.abs(rel_spec) # /10

    elif specs_id[i] != "power" and rel_spec < 0:
        reward += np.abs(rel_spec)

print ("ledro reward: ", reward)



# -------- Autockt ---------


def autockt_reward(spec, goal_spec):
    """
    Reward: doesn't penalize for overshooting spec, is negative
    """
    def _lookup(spec, goal_spec):
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = (spec - goal_spec) / (goal_spec + spec)
        return norm_spec
    
    rel_specs = _lookup(spec, goal_spec)
    pos_val = []
    reward = 0.0
    for i, rel_spec in enumerate(rel_specs):
        if specs_id[i] == "ibias_max":
            rel_spec = rel_spec * -1.0  # /10.0
        if rel_spec < 0:
            reward += rel_spec
            pos_val.append(0)
        else:
            pos_val.append(1)

    return reward if reward < -0.02 else 10


print ("autockt reward: ", autockt_reward(current_specs, target_specs))
