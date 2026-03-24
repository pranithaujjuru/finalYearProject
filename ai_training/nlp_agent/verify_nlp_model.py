import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_symptoms(text: str):
    # Load our locally trained model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "../../models/nlp_agent_local")
    
    if not os.path.exists(model_dir):
        print(f"Error: Model not found at {model_dir}")
        return
        
    print(f"Loading trained NLP model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Run the model
    model.eval()
    
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(
            encoding['input_ids'], 
            attention_mask=encoding['attention_mask']
        )
        # Apply sigmoid to outputs for multi-label classification since we used BCEWithLogitsLoss
        probs = torch.sigmoid(outputs.logits)
        predictions = (probs > 0.5).int().flatten().tolist()
    
    symptom_labels = ["sciatica", "lumbar", "radiculopathy"]
    detected_symptoms = [label for pred, label in zip(predictions, symptom_labels) if pred == 1]
    
    print("\n" + "="*50)
    print("ANALYZING TRANSCRIPTION:")
    print(text[:100] + "..." if len(text) > 100 else text)
    print("\nDETECTED SYMPTOMS:")
    if detected_symptoms:
        for s in detected_symptoms:
            print(f"- {s}")
    else:
        print("None detected.")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_text_1 = "Patient presents with severe sciatica shooting down the right leg, originating from the lumbar region."
    test_text_2 = "Patient came in for a routine checkup. No pain indicated. Blood pressure normal."
    test_text_3 = "MRI shows obvious radiculopathy and possible L5 herniation."
    
    predict_symptoms(test_text_1)
    predict_symptoms(test_text_2)
    predict_symptoms(test_text_3)
