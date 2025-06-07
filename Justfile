MICROMAMBA_EXE := "./bin/micromamba"
# For Linux x86_64. Users on other systems might need to adjust the download URL.
MICROMAMBA_DOWNLOAD_URL := "https://micro.mamba.pm/api/micromamba/linux-64/latest"
EXECUTORCH_REPO_URL := "https://github.com/pytorch/executorch.git"
EXECUTORCH_BRANCH := "v0.6.0"
EXECUTORCH_CLONE_DIR := "./executorch"

ENV_NAME := "executorch_py_env"
PYTHON_VERSION := "3.10"
MAMBA_ROOT_PREFIX_VAR := "./.micromamba_root"

default: test

setup-ubuntu:
    wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
    sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list http://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list
    sudo rm -f /etc/apt/sources.list.d/lunarg-vulkan-jammy.list
    sudo apt-get update
    sudo apt-get install -y git just libvulkan1 mesa-vulkan-drivers vulkan-sdk

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

install-base-env: setup-micromamba
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} create -n {{ENV_NAME}} python={{PYTHON_VERSION}} pip git bash -c conda-forge -y || echo "Env '{{ENV_NAME}}' already exists or command failed."

# Clone and install ExecuTorch from script
install-executorch-script: install-base-env
    @echo "Cloning ExecuTorch repository (branch {{EXECUTORCH_BRANCH}})..."
    rm -rf {{EXECUTORCH_CLONE_DIR}}
    git clone --recursive --depth 1 --branch {{EXECUTORCH_BRANCH}} {{EXECUTORCH_REPO_URL}} {{EXECUTORCH_CLONE_DIR}}
    @echo "Installing ExecuTorch requirements using install_requirements.sh..."
    (cd {{EXECUTORCH_CLONE_DIR}} && MAMBA_ROOT_PREFIX=../{{MAMBA_ROOT_PREFIX_VAR}} ../bin/micromamba run -n {{ENV_NAME}} bash ./install_requirements.sh)
    @echo "Installing ExecuTorch using install_executorch.sh..."
    (cd {{EXECUTORCH_CLONE_DIR}} && MAMBA_ROOT_PREFIX=../{{MAMBA_ROOT_PREFIX_VAR}} ../bin/micromamba run -n {{ENV_NAME}} bash ./install_executorch.sh --pybind vulkan,xnnpack)

install-deps: install-executorch-script
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} pip install --upgrade pip
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} pip install zstd certifi torch torchvision torchaudio scikit-learn matplotlib numpy

test: install-deps
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python export_multi_backend.py
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python benchmark.py

test-training: install-deps
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python train.py
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python export_trained_model.py
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python benchmark_trained.py

clean:
    rm -f linear_regression/models/*.pte
    rm -rf {{EXECUTORCH_CLONE_DIR}}
    # To also clean micromamba installation and environments, you can add:
    # echo "To remove micromamba environment and executable, run: rm -rf {{MAMBA_ROOT_PREFIX_VAR}} ./bin"
