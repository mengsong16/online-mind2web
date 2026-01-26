"""Evaluation configuration.

This file centralizes per-model max token budgets so you can tweak them
without hunting through multiple code paths.

You can also override them via environment variables when debugging, e.g.:
  KEY_POINT_MAX_TOKENS=2048 SCORE_MAX_TOKENS=1024 JUDGE_MAX_TOKENS=4096 python run.py ...
"""

# Max tokens for the Key-Point model (identify_key_points)
KEY_POINT_MAX_TOKENS = 512

# Max tokens for the Score model (judge_image -> outputs Score: 1-5)
SCORE_MAX_TOKENS = 512

# Max tokens for the final Judge model (outputs Status: "success"/"failure")
JUDGE_MAX_TOKENS = 512
