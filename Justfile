MICROMAMBA_EXE := "./bin/micromamba"
MICROMAMBA_DOWNLOAD_URL := "https://micro.mamba.pm/api/micromamba/linux-64/latest"

ENV_NAME := "executorch_py_env"
PYTHON_VERSION := "3.10"
MAMBA_ROOT_PREFIX_VAR := "./.micromamba_root"

# Path to the executorch repository.
# Assumes 'executorch' is a sibling to this 'executorch-examples' project,
# or path is overridden by EXECUTORCH_ROOT environment variable.
EXECUTORCH_ROOT_DIR := env("EXECUTORCH_ROOT", "../executorch")

default: test

setup-ubuntu:
    wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
    sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list http://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list
    sudo rm -f /etc/apt/sources.list.d/lunarg-vulkan-jammy.list
    sudo apt-get update
    sudo apt-get install -y git just libvulkan1 mesa-vulkan-drivers vulkan-sdk gcc-10 g++-10
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100
    gcc --version
    g++ --version

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
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} create -n {{ENV_NAME}} python={{PYTHON_VERSION}} pip git bash coreutils -c conda-forge -y || echo "Env '{{ENV_NAME}}' already exists or command failed."

install-deps: install-base-env
    @echo "Installing Python dependencies including ExecuTorch via pip..."
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} pip install --upgrade pip
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} pip install executorch
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} pip install zstd certifi scikit-learn matplotlib numpy

test-linear-regression: install-deps
    @echo "Listing installed executorch files..."
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} bash -c 'INSTALLED_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])"); ET_PKG_DIR=$INSTALLED_SITE_PACKAGES/executorch; if [ -d "$ET_PKG_DIR" ]; then ls -R "$ET_PKG_DIR"; else echo "Directory $ET_PKG_DIR does not exist."; fi'
    @echo "Running linear regression benchmark..."
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python benchmark.py

test: test-linear-regression
    @echo "All main tests completed."

test-training: install-deps
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python train.py
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python export_trained_model.py
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python benchmark_trained.py

clean:
    rm -f models/*.pte
    rm -f linear_regression/models/*.pte
    # To also clean micromamba installation and environments, you can add:
    # echo "To remove micromamba environment and executable, run: rm -rf {{MAMBA_ROOT_PREFIX_VAR}} ./bin"
