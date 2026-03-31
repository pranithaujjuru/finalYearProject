import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def generate_extraction(text, model, tokenizer):
    """
    Generate structured medical JSON using the 'Accuracy-First' prompt template.
    """
    prompt = f"### Instruction:\nYou are a Senior Clinical Document Specialist. Extract the relevant medical findings from the patient input into a structured JSON format.\n\n### Input:\n{text}\n\n### Output:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1, # Low temperature for more deterministic/accurate JSON
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the content after '### Output:'
    if "### Output:" in response:
        return response.split("### Output:")[1].strip()
    return response

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = os.path.join(base_dir, "../../models/nlp_agent_sft/final_adapter")
    
    print(f"Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    if os.path.exists(adapter_path):
        print(f"Loading trained LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("⚠️ No adapter found. Using base model for testing.")
        model = base_model

    # Test cases
    test_reports = [
        "Patient presents with severe lower back pain radiating down the left leg. History of L4-L5 disc bulge mentioned in 2022.",
        "65-year-old female complaining of numbness in both feet and difficulty walking more than 100 meters. Likely spinal stenosis."
    ]

    print("\n--- Starting Inference Test ---")
    for i, report in enumerate(test_reports):
        print(f"\n[Test {i+1}] Input: {report}")
        result = generate_extraction(report, model, tokenizer)
        print(f"[Test {i+1}] Extracted JSON:\n{result}")

if __name__ == "__main__":
    main()
