import subprocess

import torch


def main() -> None:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1024**3:.2f} GB)")

    try:
        print(subprocess.check_output(["nvidia-smi"], text=True))
    except Exception as e:
        print("nvidia-smi unavailable:", e)


if __name__ == "__main__":
    main()
