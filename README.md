
# Reinforcement Learning for Dynamic Tax Policy Optimization under Economic Uncertainty

**Repository:** `rl-tax-policy-optimization`

Adaptive tax policy with reinforcement learning in a stylized macroeconomic environment. The agent tunes **personal** and **corporate** tax rates to balance growth, inequality, and fiscal sustainability under uncertainty.

## Quickstart
```bash
pip install -r requirements.txt
python src/train.py --algo PPO --timesteps 300000
python src/evaluate.py --model_path results/models/ppo_tax_policy.zip --episodes 10 --plot
```

## Structure
- `src/environment.py` — Gymnasium environment `TaxEnv`
- `src/agent.py` — Model factory for PPO/A2C/SAC (SB3)
- `src/train.py` — Training entrypoint with eval/checkpoints
- `src/evaluate.py` — Rollouts + KPI plots
- `docs/` — Architecture, methodology, experiments, references
- `data/` — Synthetic macro series and sample sim outputs
- `results/` — Models and plots (created at runtime)
