import os
import subprocess
import sys

def main():
    # Create models directory, relative to the new script's location in llm/
    # So, ../models will be at executorch-examples/models/
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)

    output_file = os.path.join(models_dir, "qwen3-0_6b.pte")
    
    # Determine project root (executorch-examples directory)
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Determine the path to the main 'executorch' repository.
    # This value is expected to be relative to 'project_root'.
    # It's passed from the Justfile via EXECUTORCH_ROOT_FOR_SCRIPT,
    # defaulting to "../executorch" if not set (e.g., when script is run manually).
    executorch_repo_relative_path = os.environ.get("EXECUTORCH_ROOT_FOR_SCRIPT", "../executorch")
    executorch_repo_root_abs = os.path.abspath(os.path.join(project_root, executorch_repo_relative_path))

    # Verify that the examples directory exists in the determined executorch_repo_root_abs
    examples_dir_in_executorch_repo = os.path.join(executorch_repo_root_abs, "examples")
    if not os.path.isdir(examples_dir_in_executorch_repo):
        print(f"❌ Error: 'examples' directory not found in the determined ExecuTorch repository path: {examples_dir_in_executorch_repo}")
        print(f"   Derived from EXECUTORCH_ROOT_FOR_SCRIPT='{os.environ.get('EXECUTORCH_ROOT_FOR_SCRIPT')}' or default, resolved to '{executorch_repo_root_abs}'.")
        print(f"   Please ensure the 'executorch' repository is correctly located (e.g., as a sibling to 'executorch-examples')")
        print(f"   or the EXECUTORCH_ROOT environment variable is set correctly when running 'just'.")
        return

    # The params_file path is relative to this script's directory (llm/)
    params_file_script_relative = os.path.join("models/0_6b_config.json")
    absolute_params_file = os.path.abspath(os.path.join(script_dir, params_file_script_relative))

    print("🚀 Exporting Qwen3 0.6B model for XNNPACK backend...")
    print(f"Using params file (resolved): {absolute_params_file}")
    # Output file path is relative to the models_dir, which is relative to project_root for the subprocess
    output_file_project_relative = os.path.join("models", "qwen3-0_6b.pte")
    print(f"Outputting to (project relative for subprocess): {output_file_project_relative}")

    if not os.path.exists(absolute_params_file):
        print(f"❌ Error: Parameters file not found at {absolute_params_file}")
        print("Please ensure the path is correct and the executorch repository is cloned as expected.")
        return

    command = [
        sys.executable,  # Use the python from the current environment
        "-m",
        "examples.models.llama.export_llama",
        "--model",
        "qwen3-0_6b",
        "--params",
        # Use the relative path from the project_root for the --params argument
        os.path.join("llm/models/0_6b_config.json"),
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
        output_file_project_relative, # Use project-relative path for output
        "--verbose",
    ]

    try:
        print(f"Executing command: {' '.join(command)}")
        
        # Setting cwd to project root.
        print(f"Running subprocess with CWD: {project_root}")

        # Prepare environment for the subprocess
        current_env = os.environ.copy()
        # Prepend the executorch_repo_root_abs to PYTHONPATH
        # This allows `python -m examples.models.llama.export_llama` to find the module
        current_env["PYTHONPATH"] = f"{executorch_repo_root_abs}{os.pathsep}{current_env.get('PYTHONPATH', '')}"
        print(f"Augmented PYTHONPATH for subprocess: {current_env['PYTHONPATH']}")

        process = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True, 
            cwd=project_root,
            env=current_env # Pass the modified environment
        )
        print("✅ Qwen3 0.6B model exported successfully!")
        # Output path is now relative to project_root, so construct it from there.
        final_output_path = os.path.join(project_root, output_file_project_relative)
        print(f"Output written to {final_output_path}") 
        if process.stdout:
            print("Stdout:\n", process.stdout)
        if process.stderr:
            print("Stderr:\n", process.stderr)

    except subprocess.CalledProcessError as e:
        print("❌ Failed to export Qwen3 0.6B model.")
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
