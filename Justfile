MICROMAMBA_EXE := "./bin/micromamba"
# For Linux x86_64. Users on other systems might need to adjust the download URL.
MICROMAMBA_DOWNLOAD_URL := "https://micro.mamba.pm/api/micromamba/linux-64/latest"

ENV_NAME := "executorch_py_env"
PYTHON_VERSION := "3.10"
MAMBA_ROOT_PREFIX_VAR := "./.micromamba_root"

default: test

setup-ubuntu:
    sudo apt-get update && sudo apt-get install -y just libvulkan1 mesa-vulkan-drivers vulkan-utils

# Download and setup micromamba executable
setup-micromamba:
    @if [ ! -f "{{MICROMAMBA_EXE}}" ]; then \
        echo "Downloading micromamba to {{MICROMAMBA_EXE}}..."; \
        mkdir -p $(dirname {{MICROMAMBA_EXE}}); \
        curl -Ls {{MICROMAMBA_DOWNLOAD_URL}} | tar -xvj bin/micromamba; \
        chmod +x {{MICROMAMBA_EXE}}; \
    else \
        echo "Micromamba already installed at {{MICROMAMBA_EXE}}"; \
    fi

install-deps: setup-micromamba
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} create -n {{ENV_NAME}} python={{PYTHON_VERSION}} pip -c conda-forge -y || echo "Env '{{ENV_NAME}}' already exists or command failed."
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} pip install --upgrade pip
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} pip install zstd certifi torch torchvision torchaudio scikit-learn matplotlib numpy executorch

test: install-deps
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python export_multi_backend.py
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python benchmark.py

test-training: install-deps
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python train.py
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python export_trained_model.py
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python benchmark_trained.py

clean:
    rm -f linear_regression/models/*.pte
    # To also clean micromamba installation and environments, you can add:
    # echo "To remove micromamba environment and executable, run: rm -rf {{MAMBA_ROOT_PREFIX_VAR}} ./bin"
