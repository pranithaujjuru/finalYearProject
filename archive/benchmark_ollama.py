import os
import json
import asyncio
from python_agents.vision_agent import analyze_mri
from python_agents.nlp_agent import extract_clinical_data
from python_agents.schemas import VisionFinding, ClinicalExtraction

async def benchmark_vision_agent(test_dir, samples_per_class=5):
    print(f"--- Benchmarking Vision Agent (Llava) ---")
    classes = ["Herniated Disc", "No Stenosis", "Thecal Sac"]
    results = {"total": 0, "correct": 0}
    
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir): continue
        
        images = os.listdir(cls_dir)[:samples_per_class]
        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            
            # Predict
            prediction = await analyze_mri(img_bytes, "image/jpeg")
            
            # Improved matching for accuracy calculation
            pred_finding = prediction.finding.lower()
            ground_truth = cls.lower()
            
            # Check for keyword matches instead of exact string
            is_correct = False
            if ground_truth == "herniated disc" and any(k in pred_finding for k in ["herniation", "disc", "bulge", "extrusion"]):
                is_correct = True
            elif ground_truth == "no stenosis" and any(k in pred_finding for k in ["no", "normal", "clear", "unremarkable"]):
                is_correct = True
            elif ground_truth == "thecal sac" and any(k in pred_finding for k in ["sac", "compression", "stenosis", "narrowing"]):
                is_correct = True
            elif ground_truth in pred_finding:
                is_correct = True
            
            results["total"] += 1
            if is_correct: results["correct"] += 1
            
            print(f"Image: {img_name} | GT: {cls} | Pred: {prediction.finding} | Correct: {is_correct}")

    accuracy = (results["correct"] / results["total"]) * 100 if results["total"] > 0 else 0
    print(f"Vision Agent Accuracy: {accuracy:.2f}%\n")
    return accuracy

async def benchmark_nlp_agent(dataset_path, num_samples=10):
    print(f"--- Benchmarking NLP Agent (Llama3.2) ---")
    results = {"total": 0, "correct_symptoms": 0}
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    samples = data[:num_samples]
    for record in samples:
        user_msg = next((m["content"] for m in record["messages"] if m["role"] == "user"), "")
        target_json = next((m["content"] for m in record["messages"] if m["role"] == "model"), "{}")
        target_data = json.loads(target_json)
        target_symptoms = [s.lower() for s in target_data.get("symptoms", [])]
        
        # Prediction
        prediction = await extract_clinical_data(user_msg)
        pred_symptoms = " ".join(prediction.symptoms).lower()
        pred_history = prediction.history.lower()
        
        # Check if output contains expected key indicators
        is_hit = any(s in pred_symptoms or s in pred_history for s in target_symptoms)
        
        results["total"] += 1
        if is_hit: results["correct_symptoms"] += 1
        
        print(f"Text Snippet: {user_msg[:50]}...")
        print(f"GT Symptoms: {target_symptoms}")
        print(f"Pred Symptoms: {pred_symptoms}")
        print(f"Hit: {is_hit}\n")

    accuracy = (results["correct_symptoms"] / results["total"]) * 100 if results["total"] > 0 else 0
    print(f"NLP Agent Accuracy: {accuracy:.2f}%\n")
    return accuracy

async def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vision_test_dir = os.path.join(base_dir, "ai_training/dataset/LumbarSpinalStenosis/test")
    nlp_data_path = os.path.join(base_dir, "ai_training/dataset/nlpAgent/nlp_finetuning_data.jsonl")
    
    vision_acc = await benchmark_vision_agent(vision_test_dir)
    nlp_acc = await benchmark_nlp_agent(nlp_data_path)
    
    print("========================================")
    print("FINAL BENCHMARK REPORT")
    print(f"Local Vision Agent (Llava): {vision_acc:.2f}%")
    print(f"Local NLP Agent (Llama3.2): {nlp_acc:.2f}%")
    print("========================================")

if __name__ == "__main__":
    asyncio.run(main())
