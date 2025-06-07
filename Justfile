MICROMAMBA_EXE := "./bin/micromamba"
MICROMAMBA_DOWNLOAD_URL := "https://micro.mamba.pm/api/micromamba/linux-64/latest"

ENV_NAME := "executorch_py_env"
PYTHON_VERSION := "3.10"
MAMBA_ROOT_PREFIX_VAR := "./.micromamba_root"

# Path to the executorch repository.
# Assumes 'executorch' is a sibling to this 'executorch-examples' project,
# or path is overridden by EXECUTORCH_ROOT environment variable.
EXECUTORCH_ROOT_DIR := env("EXECUTORCH_ROOT", "../executorch")

_HF_QWEN3_0_6B_MODEL_NAME_FOR_HUB := "Qwen--Qwen3-0.6B"
_HF_QWEN3_0_6B_SNAPSHOT := "a9c98e602b9d36d2a2f7ba1eb0f5f31e4e8e5143"
_HF_TOKENIZER_BASE_PATH := "~/.cache/huggingface/hub/models--{{_HF_QWEN3_0_6B_MODEL_NAME_FOR_HUB}}/snapshots/{{_HF_QWEN3_0_6B_SNAPSHOT}}"

QWEN3_0_6B_TOKENIZER_JSON := _HF_TOKENIZER_BASE_PATH + "/tokenizer.json"
QWEN3_0_6B_TOKENIZER_CONFIG_JSON := _HF_TOKENIZER_BASE_PATH + "/tokenizer_config.json"
QWEN3_0_6B_PARAMS_FILE := "./llm/models/0_6b_config.json"

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

export-qwen3-0_6b: install-deps
    @echo "Exporting Qwen3 0.6B model..."
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} python llm/export_qwen3_0_6b.py

export-qwen3-1_7b: install-deps
    @echo "Exporting Qwen3 1.7B model..."
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} python llm/export_qwen3_1_7b.py

export-qwen3-4b: install-deps
    @echo "Exporting Qwen3 4B model..."
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} python llm/export_qwen3_4b.py

export-llms: export-qwen3-0_6b export-qwen3-1_7b export-qwen3-4b
    @echo "All Qwen3 LLM models exported."

test-linear-regression: install-deps
    @echo "Listing installed executorch files..."
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} bash -c 'INSTALLED_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])"); ET_PKG_DIR=$INSTALLED_SITE_PACKAGES/executorch; if [ -d "$ET_PKG_DIR" ]; then ls -R "$ET_PKG_DIR"; else echo "Directory $ET_PKG_DIR does not exist."; fi'
    @echo "Running linear regression benchmark..."
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd linear_regression/python python benchmark.py

test-llm-qwen3-0_6b: export-qwen3-0_6b
    @echo "Running Qwen3 0.6B LLM example..."
    @echo "Verifying required files for LLM run:"
    @if [ ! -f "models/qwen3-0_6b.pte" ]; then echo "Error: PTE file models/qwen3-0_6b.pte not found. Run 'just export-qwen3-0_6b' first."; exit 1; fi
    @if [ ! -f "{{QWEN3_0_6B_PARAMS_FILE}}" ]; then echo "Error: Params file {{QWEN3_0_6B_PARAMS_FILE}} not found. Check path (currently '{{QWEN3_0_6B_PARAMS_FILE}}'). Ensure it exists at ./llm/models/0_6b_config.json"; exit 1; fi
    @echo "Attempting to run LLM. If tokenizer files are missing from HF cache ({{_HF_TOKENIZER_BASE_PATH}}), the script will fail."
    MAMBA_ROOT_PREFIX={{MAMBA_ROOT_PREFIX_VAR}} {{MICROMAMBA_EXE}} run -n {{ENV_NAME}} --cwd . python -m examples.models.llama.runner.native \
        --model qwen3-0_6b \
        --pte models/qwen3-0_6b.pte \
        --tokenizer "{{QWEN3_0_6B_TOKENIZER_JSON}}" \
        --tokenizer_config "{{QWEN3_0_6B_TOKENIZER_CONFIG_JSON}}" \
        --prompt "Who is the president of the US?" \
        --params "{{QWEN3_0_6B_PARAMS_FILE}}" \
        --max_len 128 \
        -kv \
        --temperature 0.6

test: test-linear-regression test-llm-qwen3-0_6b
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
