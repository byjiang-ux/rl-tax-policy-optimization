
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from .utils import clamp
from .config import EnvConfig, RewardWeights

@dataclass
class State:
    gdp: float
    unemp: float
    infl: float
    gini: float
    deficit_ratio: float
    debt_ratio: float
    rate: float
    output_gap: float

class TaxEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_cfg: EnvConfig = EnvConfig(), rew: RewardWeights = RewardWeights(), seed: int = 42):
        super().__init__()
        self.env_cfg = env_cfg
        self.rew = rew
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-env_cfg.max_action_delta, high=env_cfg.max_action_delta, shape=(2,), dtype=np.float32)
        self._reset_internal()

    def _reset_internal(self):
        self.t = 0
        self.personal_tax = self.env_cfg.personal_tax_init
        self.corporate_tax = self.env_cfg.corporate_tax_init
        self.state = State(
            gdp=100.0 + self.rng.normal(0, 1),
            unemp=np.clip(5 + self.rng.normal(0,1), 3, 12),
            infl=np.clip(2 + self.rng.normal(0,0.5), -2, 8) * 100/100,  # percentage points
            gini=np.clip(0.36 + self.rng.normal(0,0.01), 0.30, 0.50),
            deficit_ratio=np.clip(self.rng.normal(0.03,0.02), -0.05, 0.12),
            debt_ratio=np.clip(0.6 + self.rng.normal(0,0.05), 0.30, 1.2),
            rate=np.clip(2 + self.rng.normal(0,1), 0, 10),
            output_gap=np.clip(self.rng.normal(0,1), -5, 5)
        )
        self.prev_gdp = self.state.gdp
        self.prev_personal_tax = self.personal_tax
        self.prev_corporate_tax = self.corporate_tax

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_internal()
        return self._obs(), {}

    def _obs(self):
        s = self.state
        return np.array([s.gdp, s.unemp, s.infl, s.gini, s.deficit_ratio, s.debt_ratio, s.rate, s.output_gap], dtype=np.float32)

    def step(self, action):
        d_pt, d_ct = float(action[0]), float(action[1])
        self.personal_tax = clamp(self.personal_tax + d_pt, self.env_cfg.tax_min, self.env_cfg.tax_max)
        self.corporate_tax = clamp(self.corporate_tax + d_ct, self.env_cfg.tax_min, self.env_cfg.tax_max)

        s = self.state
        tax_burden = 0.5*self.personal_tax + 0.5*self.corporate_tax
        demand_effect = -0.8 * (self.personal_tax - 0.25)
        invest_effect = -1.0 * (self.corporate_tax - 0.22)
        shock_regime = self.rng.choice([0.6, 0.1, -0.4], p=[0.6,0.25,0.15])
        epsilon = self.rng.normal(0, 0.7)

        gdp_growth = 0.3* (s.output_gap/10.0) + demand_effect + invest_effect + shock_regime + 0.2*epsilon
        new_gdp = max(1.0, s.gdp * (1 + gdp_growth/100.0))

        new_unemp = np.clip(s.unemp - 0.3*(gdp_growth) + self.rng.normal(0,0.15), 2.5, 20.0)
        infl_gap = (new_unemp - 4.5)
        new_infl = np.clip(s.infl + (-0.2*infl_gap) + 0.3*self.rng.normal(0,1) + 0.1*(tax_burden-0.25)*100, -2, 15)
        new_gini = np.clip(s.gini + 0.002*(s.output_gap/5.0) - 0.05*(self.personal_tax - 0.25), 0.25, 0.6)

        revenue_ratio = np.clip(0.15 + 0.5*tax_burden - 0.1*max(gdp_growth, 0), 0.05, 0.5)
        spend_ratio = 0.18 + 0.04*np.clip( (new_unemp-4.0)/10.0, 0, 1)
        new_deficit = np.clip(spend_ratio - revenue_ratio, -0.05, 0.15)
        new_debt = np.clip(s.debt_ratio*(1 + new_deficit) + 0.01*(s.rate/100.0), 0.1, 2.5)

        new_rate = np.clip(1.0 + 1.2*(new_infl - 2.0) + 0.5*(s.output_gap/2.0), 0, 15)
        new_gap = np.clip(0.8*s.output_gap + 0.5*(gdp_growth) + self.rng.normal(0,0.6), -6, 6)

        self.prev_gdp = s.gdp
        self.prev_personal_tax = self.personal_tax
        self.prev_corporate_tax = self.corporate_tax
        self.state = State(new_gdp, new_unemp, new_infl, new_gini, new_deficit, new_debt, new_rate, new_gap)

        growth = (new_gdp - self.prev_gdp)/max(1e-6, self.prev_gdp) * 100.0
        infl_dev = abs(new_infl - self.env_cfg.inflation_target*100)
        deficit_pen = (new_deficit)**2
        debt_pen = max(0.0, new_debt - self.env_cfg.debt_limit)**2
        tax_level_pen = (self.personal_tax + self.corporate_tax)/2.0
        tax_vol = np.linalg.norm([float(self.personal_tax - self.prev_personal_tax), float(self.corporate_tax - self.prev_corporate_tax)])

        r = ( self.rew.w_growth*growth
              - self.rew.w_unemp*new_unemp
              - self.rew.w_infl*infl_dev
              - self.rew.w_gini*new_gini*100
              - self.rew.w_deficit*deficit_pen*100
              - self.rew.w_debt*debt_pen*50
              - self.rew.w_taxvol*tax_vol*100
              - self.rew.w_taxlevel*tax_level_pen*100 )

        terminated = (new_debt > 1.8) or (new_unemp >= 20.0)
        truncated = self.t >= (self.env_cfg.horizon - 1)
        self.t += 1

        info = {
            "personal_tax": self.personal_tax,
            "corporate_tax": self.corporate_tax,
            "growth_pct": growth,
            "revenue_ratio": revenue_ratio,
            "spend_ratio": spend_ratio,
        }
        return self._obs(), float(r), terminated, truncated, info

    def render(self):
        pass
