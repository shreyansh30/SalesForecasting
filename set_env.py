"""Run this once to suppress TensorFlow & oneDNN messages permanently."""
import subprocess, sys

env_var = "TF_ENABLE_ONEDNN_OPTS=0"
print(f"Setting {env_var}...")

# Windows CMD
subprocess.run(f'setx {env_var.split("=")[0]} {env_var.split("=")[1]}', shell=True)
print("Done! Restart your terminal and the TF messages will be gone.")