
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC
from .environment import TaxEnv

ALGOS = {"PPO": PPO, "A2C": A2C, "SAC": SAC}

def rollout(model, env, episodes=5, deterministic=True):
    metrics = {k: [] for k in ["reward","growth","unemp","infl","gini","debt","deficit","ptax","ctax"]}
    for _ in range(episodes):
        obs, _ = env.reset()
        done, trunc = False, False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, trunc, info = env.step(action)
            s = obs
            metrics["reward"].append(reward)
            metrics["unemp"].append(float(s[1]))
            metrics["infl"].append(float(s[2]))
            metrics["gini"].append(float(s[3]))
            metrics["debt"].append(float(s[5]))
            metrics["deficit"].append(float(s[4]))
            metrics["ptax"].append(info["personal_tax"])
            metrics["ctax"].append(info["corporate_tax"])
            metrics["growth"].append(info["growth_pct"])
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()

    env = TaxEnv(seed=123)
    algo = "PPO" if "ppo" in os.path.basename(args.model_path).lower() else "PPO"
    model = ALGOS[algo].load(args.model_path)

    metrics = rollout(model, env, episodes=args.episodes)

    print(f"Avg reward: {np.mean(metrics['reward']):.3f}")
    print(f"Avg growth %: {np.mean(metrics['growth']):.3f}")
    print(f"Avg unemployment: {np.mean(metrics['unemp']):.3f}")
    print(f"Avg inflation: {np.mean(metrics['infl']):.3f}")
    print(f"Avg debt ratio: {np.mean(metrics['debt']):.3f}")

    if args.plot:
        os.makedirs('results/plots', exist_ok=True)
        plt.figure(); plt.plot(metrics['reward']); plt.title('Reward Curve'); plt.savefig('results/plots/reward_curve.png', bbox_inches='tight')
        plt.figure(); plt.scatter(metrics['ptax'], metrics['growth'], s=8); plt.xlabel('Personal Tax'); plt.ylabel('GDP Growth (%)'); plt.title('Growth vs Personal Tax'); plt.savefig('results/plots/gdp_vs_taxrate.png', bbox_inches='tight')
        plt.figure(); plt.scatter(metrics['ptax'], metrics['gini'], s=8); plt.xlabel('Personal Tax'); plt.ylabel('Gini'); plt.title('Inequality vs Personal Tax'); plt.savefig('results/plots/inequality_vs_policy.png', bbox_inches='tight')
        print('Plots saved to results/plots/*.png')

if __name__ == '__main__':
    main()
