
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from .environment import TaxEnv
from .config import TrainConfig

ALGOS = {"PPO": PPO, "A2C": A2C, "SAC": SAC}

def make_env(seed=42):
    return TaxEnv(seed=seed)

def make_model(algo: str, env, cfg: TrainConfig):
    algo = algo.upper()
    assert algo in ALGOS, f"Unsupported algo {algo}. Choose from {list(ALGOS.keys())}"
    Model = ALGOS[algo]
    kwargs = {"verbose": 1}
    if algo in ["PPO", "A2C"]:
        kwargs.update(dict(learning_rate=cfg.lr, gamma=cfg.gamma, seed=cfg.seed))
    if algo == "PPO":
        kwargs.update(dict(n_steps=cfg.n_steps, batch_size=cfg.batch_size, ent_coef=cfg.ent_coef))
    if algo == "SAC":
        kwargs.update(dict(learning_rate=cfg.lr, seed=cfg.seed))
    model = Model(cfg.policy, env, **kwargs)
    return model
