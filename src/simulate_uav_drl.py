import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

GRID_SIZE = 20
EPISODES = 120
STEPS = 150
START_BATTERY = 100.0

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
    4: (0, 0),   # hover
}

def run_policy(policy_name: str):
    episode_rows = []

    for ep in range(EPISODES):
        pos = np.array([GRID_SIZE // 2, GRID_SIZE // 2])
        battery = START_BATTERY
        visited = set()
        reward_total = 0.0

        for step in range(STEPS):
            if battery <= 0:
                break

            if policy_name == "random":
                action = np.random.choice(list(ACTIONS.keys()))
            elif policy_name == "greedy":
                # move toward nearest unvisited cell by simple directional heuristic
                unvisited = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if (i, j) not in visited]
                if unvisited:
                    target = np.array(unvisited[np.random.randint(len(unvisited))])
                    diff = target - pos
                    if abs(diff[0]) > abs(diff[1]):
                        action = 1 if diff[0] > 0 else 0
                    else:
                        action = 3 if diff[1] > 0 else 2
                else:
                    action = 4
            else:
                # DRL-inspired energy-aware policy:
                # favors unexplored high-value movements, avoids boundaries and low-battery waste
                candidates = []
                for a, move in ACTIONS.items():
                    nxt = np.clip(pos + np.array(move), 0, GRID_SIZE - 1)
                    new_cell_bonus = 2.0 if tuple(nxt) not in visited else -0.25
                    boundary_penalty = -0.5 if np.any(nxt == 0) or np.any(nxt == GRID_SIZE - 1) else 0
                    energy_penalty = -0.2 if a != 4 else -0.05
                    low_battery_penalty = -0.5 if battery < 20 and a != 4 else 0
                    score = new_cell_bonus + boundary_penalty + energy_penalty + low_battery_penalty + np.random.normal(0, 0.08)
                    candidates.append((score, a))
                action = max(candidates)[1]

            move = np.array(ACTIONS[action])
            new_pos = np.clip(pos + move, 0, GRID_SIZE - 1)
            moved = not np.array_equal(new_pos, pos)
            pos = new_pos

            energy_cost = 0.8 if moved else 0.25
            battery -= energy_cost

            before = len(visited)
            visited.add(tuple(pos))
            coverage_gain = len(visited) - before

            reward = coverage_gain * 3.0 - energy_cost * 0.6
            reward_total += reward

        coverage_ratio = len(visited) / (GRID_SIZE * GRID_SIZE)
        energy_used = START_BATTERY - max(battery, 0)

        episode_rows.append({
            "policy": policy_name,
            "episode": ep + 1,
            "coverage_ratio": coverage_ratio,
            "energy_used": energy_used,
            "battery_remaining": max(battery, 0),
            "steps_completed": step + 1,
            "reward": reward_total,
            "coverage_per_energy": coverage_ratio / max(energy_used, 1e-6),
        })

    return pd.DataFrame(episode_rows)

df = pd.concat([
    run_policy("random"),
    run_policy("greedy"),
    run_policy("drl_energy_aware")
], ignore_index=True)

df.to_csv(OUTPUT / "episode_results.csv", index=False)

summary = df.groupby("policy").agg({
    "coverage_ratio": ["mean", "std"],
    "energy_used": ["mean", "std"],
    "battery_remaining": ["mean", "std"],
    "steps_completed": ["mean", "std"],
    "reward": ["mean", "std"],
    "coverage_per_energy": ["mean", "std"],
}).round(4)

summary.to_csv(OUTPUT / "summary_table.csv")

for metric, ylabel, filename in [
    ("coverage_ratio", "Coverage Ratio", "coverage_comparison.png"),
    ("energy_used", "Energy Used", "energy_comparison.png"),
    ("reward", "Episode Reward", "reward_comparison.png"),
    ("coverage_per_energy", "Coverage per Energy Unit", "sustainability_efficiency.png"),
]:
    plt.figure(figsize=(8, 5))
    for policy in df["policy"].unique():
        subset = df[df["policy"] == policy]
        rolling = subset[metric].rolling(10, min_periods=1).mean()
        plt.plot(subset["episode"], rolling, label=policy)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(ylabel + " Across Policies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=300)
    plt.close()

print("Simulation completed.")
print("Generated:")
print("- outputs/episode_results.csv")
print("- outputs/summary_table.csv")
print("- outputs/coverage_comparison.png")
print("- outputs/energy_comparison.png")
print("- outputs/reward_comparison.png")
print("- outputs/sustainability_efficiency.png")
