import subprocess
import os
import sys

def run_test(name, script_path, python_exe):
    print(f"\n{'='*20} Running {name} {'='*20}")
    try:
        result = subprocess.run([python_exe, script_path], capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print("--- Standard Error ---")
            print(result.stderr)
        if result.returncode == 0:
            print(f"✅ {name} finished successfully.")
        else:
            print(f"❌ {name} failed with exit code {result.returncode}.")
    except subprocess.TimeoutExpired:
        print(f"⏰ {name} timed out after 300 seconds.")
    except Exception as e:
        print(f"🛑 Error running {name}: {e}")

def main():
    python_exe = os.path.abspath(os.path.join(os.getcwd(), 'venv_new', 'Scripts', 'python.exe'))
    if not os.path.exists(python_exe):
        python_exe = sys.executable # Fallback to current python if venv not ready
        print(f"⚠️ venv_new not ready, using {python_exe}")

    tests = [
        ("NLP Agent", "tmp/test_nlp_agent.py"),
        ("Vision Agent", "tmp/test_vision_agent.py"),
        ("Validator Agent", "tmp/test_validator_agent.py"),
        ("Translator Agent", "tmp/test_translator_agent.py"),
        ("Main Pipeline", "main_pipeline.py")
    ]

    for name, path in tests:
        run_test(name, path, python_exe)

if __name__ == "__main__":
    main()
