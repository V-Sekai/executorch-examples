import os
import subprocess
import sys

def main():
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)

    output_file = os.path.join(models_dir, "qwen3-4b.pte")
    
    params_file = os.path.join(
        "models/4b_config.json"  # Updated path
    )

    print("🚀 Exporting Qwen3 4B model for XNNPACK backend...")
    print(f"Using params file: {params_file}")
    print(f"Outputting to: {output_file}")

    script_dir = os.path.dirname(__file__)
    absolute_params_file = os.path.abspath(os.path.join(script_dir, params_file))

    if not os.path.exists(absolute_params_file):
        print(f"❌ Error: Parameters file not found at {absolute_params_file}")
        print("Please ensure the path is correct and the executorch repository is cloned as expected.")
        return

    command = [
        sys.executable,
        "-m",
        "examples.models.llama.export_llama",
        "--model",
        "qwen3-4b",
        "--params",
        os.path.join("llm/models/4b_config.json"),  # Updated path
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
        os.path.join("models", "qwen3-4b.pte"), 
        "--verbose",
    ]

    try:
        print(f"Executing command: {' '.join(command)}")
        project_root = os.path.abspath(os.path.join(script_dir, "../"))
        print(f"Running subprocess with CWD: {project_root}")

        process = subprocess.run(
            command, check=True, capture_output=True, text=True, cwd=project_root
        )
        print("✅ Qwen3 4B model exported successfully!")
        final_output_path = os.path.join(project_root, "models", "qwen3-4b.pte")
        print(f"Output written to {final_output_path}") 
        if process.stdout:
            print("Stdout:\n", process.stdout)
        if process.stderr:
            print("Stderr:\n", process.stderr)

    except subprocess.CalledProcessError as e:
        print("❌ Failed to export Qwen3 4B model.")
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
