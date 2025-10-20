
import argparse, os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from .agent import make_env, make_model
from .config import TrainConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', default='PPO')
    p.add_argument('--timesteps', type=int, default=300000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    cfg = TrainConfig(algo=args.algo, total_timesteps=args.timesteps, seed=args.seed)
    env = DummyVecEnv([lambda: make_env(seed=args.seed)])
    model = make_model(cfg.algo, env, cfg)

    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    eval_env = DummyVecEnv([lambda: make_env(seed=args.seed+1)])
    eval_cb = EvalCallback(eval_env, best_model_save_path='results/models', log_path='results/models', eval_freq=5000, n_eval_episodes=5, deterministic=True)
    ckpt_cb = CheckpointCallback(save_freq=10000, save_path='results/models', name_prefix=f'{cfg.algo.lower()}_tax_policy_ckpt')

    model.learn(total_timesteps=cfg.total_timesteps, callback=[eval_cb, ckpt_cb])
    model.save('results/models/ppo_tax_policy.zip')
    print('Training complete. Model saved to results/models/ppo_tax_policy.zip')

if __name__ == '__main__':
    main()
