import subprocess
import os
import sys
import time

def run_command(command, description):
    print(f"\n{'='*50}")
    print(f"STARTING: {description}")
    print(f"{'='*50}")
    
    # Use the specific venv python
    venv_python = os.path.join("venv_new", "Scripts", "python.exe")
    full_command = [venv_python] + command
    
    process = subprocess.Popen(
        full_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream output to console
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode == 0:
        print(f"\nSUCCESS: {description} completed.")
    else:
        print(f"\nFAILURE: {description} failed with exit code {process.returncode}.")
    return process.returncode

def main():
    print("🌙 INITIALIZING OVERNIGHT TRAINING PIPELINE...")
    
    # 1. Vision Agent Training (The heavy one)
    vision_task = ["training/vision_agent/train_vision_agent.py"]
    ret_vision = run_command(vision_task, "Swin-Large Vision Agent Fine-Tuning")
    
    if ret_vision != 0:
        print("Stopping due to Vision Agent failure.")
        sys.exit(1)
        
    # 2. NLP Agent Training
    nlp_task = ["training/nlp_agent/train_nlp_agent.py"]
    ret_nlp = run_command(nlp_task, "Qwen-LoRA NLP Agent Fine-Tuning")
    
    if ret_nlp != 0:
        print("Stopping due to NLP Agent failure.")
        sys.exit(1)
        
    print("\n✅ OVERNIGHT TRAINING PIPELINE COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
