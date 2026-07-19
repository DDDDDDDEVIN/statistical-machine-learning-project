"""Reusable code extracted from the project notebooks.

The notebooks cluster into three families that share their data pipeline and
model definitions within each family, but diverge across families:

- ``baselines`` : bert_baseline, mlp_baseline
- ``specific``  : specific_lstm, specific_mlp
- ``semi``      : MAIN_unified_SMOTE_semi, unified_mixup_semi

Truly universal pieces (identical across all six notebooks) live in
``common``; each family module imports from it.
"""
