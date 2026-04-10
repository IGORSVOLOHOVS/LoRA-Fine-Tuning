import os
import yaml
import subprocess
import sys

def open_tensorboard():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "train_config_v2.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output", {}).get("output_dir", "./checkpoints")
    log_dir = os.path.normpath(os.path.join(os.path.dirname(config_path), "..", output_dir, "logs"))

    if not os.path.exists(log_dir):
        print(f"Warning: Log directory {log_dir} does not exist yet. It will be created when training starts.")

    print(f"Launching TensorBoard for log directory: {log_dir}")
    
    try:
        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    open_tensorboard()
