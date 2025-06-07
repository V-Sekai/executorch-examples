import os
import subprocess
import sys

def main():
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)

    output_file = os.path.join(models_dir, "qwen3-1_7b.pte")
    
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    executorch_repo_relative_path = os.environ.get("EXECUTORCH_ROOT_FOR_SCRIPT", "../executorch")
    executorch_repo_root_abs = os.path.abspath(os.path.join(project_root, executorch_repo_relative_path))

    examples_dir_in_executorch_repo = os.path.join(executorch_repo_root_abs, "examples")
    if not os.path.isdir(examples_dir_in_executorch_repo):
        print(f"❌ Error: 'examples' directory not found in the determined ExecuTorch repository path: {examples_dir_in_executorch_repo}")
        print(f"   Derived from EXECUTORCH_ROOT_FOR_SCRIPT='{os.environ.get('EXECUTORCH_ROOT_FOR_SCRIPT')}' or default, resolved to '{executorch_repo_root_abs}'.")
        print(f"   Please ensure the 'executorch' repository is correctly located (e.g., as a sibling to 'executorch-examples')")
        print(f"   or the EXECUTORCH_ROOT environment variable is set correctly when running 'just'.")
        return

    params_file_script_relative = os.path.join("models/1_7b_config.json")
    absolute_params_file = os.path.abspath(os.path.join(script_dir, params_file_script_relative))

    print("🚀 Exporting Qwen3 1.7B model for XNNPACK backend...")
    print(f"Using params file (resolved): {absolute_params_file}")
    output_file_project_relative = os.path.join("models", "qwen3-1_7b.pte")
    print(f"Outputting to (project relative for subprocess): {output_file_project_relative}")

    if not os.path.exists(absolute_params_file):
        print(f"❌ Error: Parameters file not found at {absolute_params_file}")
        print("Please ensure the path is correct and the executorch repository is cloned as expected.")
        return

    command = [
        sys.executable,
        "-m",
        "examples.models.llama.export_llama",
        "--model",
        "qwen3-1_7b",
        "--params",
        os.path.join("llm/models/1_7b_config.json"),
        "-kv",
        "--use_sdpa_with_kv_cache",
        "-d",
        "fp32",
        "-X",
        "--xnnpack-extended-ops",
        "-qmode",
        "8da4w",
        "--metadata",
        '''{"get_bos_id": 151644, "get_eos_ids":[151645]}''', 
        "--output_name",
        output_file_project_relative, 
        "--verbose",
    ]

    try:
        print(f"Executing command: {' '.join(command)}")
        print(f"Running subprocess with CWD: {project_root}")

        current_env = os.environ.copy()
        current_env["PYTHONPATH"] = f"{executorch_repo_root_abs}{os.pathsep}{current_env.get('PYTHONPATH', '')}"
        print(f"Augmented PYTHONPATH for subprocess: {current_env['PYTHONPATH']}")

        process = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True, 
            cwd=project_root,
            env=current_env
        )
        print("✅ Qwen3 1.7B model exported successfully!")
        final_output_path = os.path.join(project_root, output_file_project_relative)
        print(f"Output written to {final_output_path}") 
        if process.stdout:
            print("Stdout:\n", process.stdout)
        if process.stderr:
            print("Stderr:\n", process.stderr)

    except subprocess.CalledProcessError as e:
        print("❌ Failed to export Qwen3 1.7B model.")
        print(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            print("Stdout:\n", e.stdout)
        if e.stderr:
            print("Stderr:\n", e.stderr)
    except FileNotFoundError:
        print(
            f"❌ Error: The python executable '{sys.executable}' or the module "
            "'examples.models.llama.export_llama' was not found."
        )
        print(
            "Ensure that ExecuTorch is correctly installed, the 'examples' module is runnable,"
        )
        print(
            "and the necessary scripts/configs are available at the expected paths."
        )
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
