ENV_NAME := "executorch_py_env"
PYTHON_VERSION := "3.10"

default: test

install-deps:
    micromamba create -n {{ENV_NAME}} python={{PYTHON_VERSION}} pip -c conda-forge -y || echo "Env '{{ENV_NAME}}' already exists or command failed."
    micromamba run -n {{ENV_NAME}} pip install --upgrade pip
    micromamba run -n {{ENV_NAME}} pip install zstd certifi torch torchvision torchaudio scikit-learn matplotlib numpy executorch

test: install-deps
    micromamba run -n {{ENV_NAME}} --cwd linear_regression/python python export_multi_backend.py
    micromamba run -n {{ENV_NAME}} --cwd linear_regression/python python benchmark.py

test-training: install-deps
    micromamba run -n {{ENV_NAME}} --cwd linear_regression/python python train.py
    micromamba run -n {{ENV_NAME}} --cwd linear_regression/python python export_trained_model.py
    micromamba run -n {{ENV_NAME}} --cwd linear_regression/python python benchmark_trained.py

clean:
    rm -f linear_regression/models/*.pte
