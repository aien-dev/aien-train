# Outlier study (L284 Part 2b), ORACLE
Source: Wgrad operands (X, dY; first 4096 tokens of a 16384-token batch) of layers b0.fc1, b2.qkv, b3.proj, b4.fc2 of the
BF16 baseline run (streams/own-fp4/train.py --recipe bf16 --seed 0 --dump_steps 200,1500,2999), data SearchQA/TinyLlama
tokens. Command: CUDA_VISIBLE_DEVICES= ~/.venv/bin/python -I streams/own-fp4/outlier_study.py RUN_DIR rows.json (CPU,
torch 2.14.0). Metric: |dW_q - dW| / |dW| (Frobenius) with dW = dY^T X in FP32; X nearest, dY stochastic (one draw).
Files: rows.json (every case, also X and dY operand errors), summary.txt (means), per_case.txt (per case).
