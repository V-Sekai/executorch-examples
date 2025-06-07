MICROMAMBA_EXE := "./bin/micromamba"
# For Linux x86_64. Users on other systems might need to adjust the download URL.
MICROMAMBA_DOWNLOAD_URL := "https://micro.mamba.pm/api/micromamba/linux-64/latest"
EXECUTORCH_REPO_URL := "https://github.com/pytorch/executorch.git"
EXECUTORCH_BRANCH := "main"
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

install-executorch-requirements: install-executorch-script
    @echo "Installing ExecuTorch Python requirements (cmake, torch, etc.)..."
    ( \
    cd {{EXECUTORCH_CLONE_DIR}} && \
    MAMBA_ROOT_PREFIX_ENV_VAR=../{{MAMBA_ROOT_PREFIX_VAR}} && \
    MICROMAMBA_EXE_PATH=../bin/micromamba && \
    ENV_PREFIX_PATH="$MAMBA_ROOT_PREFIX_ENV_VAR/envs/{{ENV_NAME}}" && \
    chmod +x ../scripts/install_executorch_requirements.sh && \
    $MICROMAMBA_EXE_PATH --root-prefix "$MAMBA_ROOT_PREFIX_ENV_VAR" run -p "$ENV_PREFIX_PATH" ../scripts/install_executorch_requirements.sh \
    )

configure-executorch-cmake: install-executorch-requirements
    @echo "Configuring ExecuTorch with CMake..."
    ( \
    cd {{EXECUTORCH_CLONE_DIR}} && \
    MAMBA_ROOT_PREFIX_ENV_VAR=../{{MAMBA_ROOT_PREFIX_VAR}} && \
    MICROMAMBA_EXE_PATH=../bin/micromamba && \
    ENV_PREFIX_PATH="$MAMBA_ROOT_PREFIX_ENV_VAR/envs/{{ENV_NAME}}" && \
    chmod +x ../scripts/configure_executorch_cmake.sh && \
    $MICROMAMBA_EXE_PATH --root-prefix "$MAMBA_ROOT_PREFIX_ENV_VAR" run -p "$ENV_PREFIX_PATH" ../scripts/configure_executorch_cmake.sh \
    )

build-executorch-cmake: configure-executorch-cmake
    @echo "Building and installing ExecuTorch with CMake..."
    ( \
    cd {{EXECUTORCH_CLONE_DIR}}/cmake-build && \
    MAMBA_ROOT_PREFIX_ENV_VAR=../../{{MAMBA_ROOT_PREFIX_VAR}} && \
    MICROMAMBA_EXE_PATH=../../bin/micromamba && \
    ENV_PREFIX_PATH="$MAMBA_ROOT_PREFIX_ENV_VAR/envs/{{ENV_NAME}}" && \
    chmod +x ../../scripts/build_executorch_cmake.sh && \
    $MICROMAMBA_EXE_PATH --root-prefix "$MAMBA_ROOT_PREFIX_ENV_VAR" run -p "$ENV_PREFIX_PATH" ../../scripts/build_executorch_cmake.sh \
    )

install-executorch-script: install-base-env
    @if [ ! -d "{{EXECUTORCH_CLONE_DIR}}/.git" ]; then \
        echo "ExecuTorch directory '{{EXECUTORCH_CLONE_DIR}}' not found or not a git repository. Cloning..."; \
        rm -rf "{{EXECUTORCH_CLONE_DIR}}"; \
        git clone --recursive --depth 1 --branch {{EXECUTORCH_BRANCH}} {{EXECUTORCH_REPO_URL}} {{EXECUTORCH_CLONE_DIR}}; \
    else \
        echo "ExecuTorch directory '{{EXECUTORCH_CLONE_DIR}}' already exists. Skipping clone."; \
    fi

install-deps: build-executorch-cmake
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
