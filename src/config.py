
from dataclasses import dataclass

@dataclass
class EnvConfig:
    horizon: int = 200
    inflation_target: float = 0.02
    debt_limit: float = 1.0
    personal_tax_init: float = 0.25
    corporate_tax_init: float = 0.22
    tax_min: float = 0.00
    tax_max: float = 0.60
    max_action_delta: float = 0.05

@dataclass
class RewardWeights:
    w_growth: float = 1.0
    w_unemp: float = 0.6
    w_infl: float = 0.5
    w_gini: float = 0.4
    w_deficit: float = 1.2
    w_debt: float = 1.0
    w_taxvol: float = 0.3
    w_taxlevel: float = 0.15

@dataclass
class TrainConfig:
    algo: str = "PPO"
    total_timesteps: int = 300_000
    policy: str = "MlpPolicy"
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 2048
    batch_size: int = 256
    ent_coef: float = 0.01
    clip_range: float = 0.2
    vf_coef: float = 0.5
    seed: int = 42
