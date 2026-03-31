import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../dataset/nlpAgent/nlp_finetuning_data.jsonl")
    output_dir = os.path.join(base_dir, "../../models/nlp_agent_sft")

    # 1. Model Configuration (GPU Friendly)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # 2. Load Model and Tokenizer (GPU Accelerated)
    print(f"Loading {model_id} with GPU acceleration...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto", 
        torch_dtype=torch.float16, 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    # 4. Load & Tokenize Dataset (Standard Trainer Pattern)
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    def tokenize_function(example):
        messages = example["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        model_msg = next((m["content"] for m in messages if m["role"] == "model"), "{}")
        
        full_text = f"### Instruction:\nYou are a Senior Clinical Document Specialist. Extract the relevant medical findings from the patient input into a structured JSON format.\n\n### Input:\n{user_msg}\n\n### Output:\n{model_msg}"
        
        tokens = tokenizer(full_text, truncation=True, max_length=512, padding="max_length")
        # For CausalLM, labels are the same as input_ids
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    print("Tokenizing dataset for standard Trainer...")
    tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        num_train_epochs=3,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        load_best_model_at_end=True,
        use_cpu=False,
        fp16=True, # Use FP16 for speed and memory efficiency
        report_to="none",
        push_to_hub=False,
    )

    # 6. Trainer Execution
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    print("Starting CPU-Optimized Accuracy-First Fine-Tuning...")
    trainer.train()

    # 7. Final Model Saving
    trainer.save_model(os.path.join(output_dir, "final_adapter"))
    print(f"Training Complete! Best model saved to {output_dir}")

if __name__ == "__main__":
    main()
