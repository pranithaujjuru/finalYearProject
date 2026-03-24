import pandas as pd
import google.generativeai as genai
import json
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Configure the Gemini API key
# Load environment variables from .env.local
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local")
load_dotenv(dotenv_path=env_path)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Define file paths
CSV_PATH = r"C:\Users\pranitha\Downloads\sciatica-prognosis-ai\ai_training\dataset\nlpAgent\mtsamples.csv"
JSONL_PATH = r"C:\Users\pranitha\Downloads\sciatica-prognosis-ai\ai_training\dataset\nlpAgent\nlp_finetuning_data.jsonl"

def extract_clinical_info(transcription: str) -> str:
    """
    Uses Gemini SDK to extract clinical info from the transcription into a strict JSON schema.
    Expected Schema: {"symptoms": [], "history": "", "suggestedProcedure": ""}
    """
    # Using 'gemini-2.5-flash' since that's what the API key provisions.
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = (
        "You are an AI medical data synthesizer. Extract the following clinical information "
        "from the provided medical transcription. You must output ONLY a valid JSON object "
        "matching this exact schema: "
        '{"symptoms": ["string", "string"], "history": "string", "suggestedProcedure": "string"}. '
        "If a specific field is not found, leave lists empty or strings blank. Do not include markdown codeblocks, just the JSON string."
        f"\n\nTranscription:\n{transcription}"
    )
    
    response = model.generate_content(prompt)
    
    # Try to clean up the response in case the model returns markdown like ```json ... ```
    content = response.text.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
        
    content = content.strip()
    
    # Optional check: verify it's valid JSON before returning. 
    # If this fails, the ValueError will be caught by the except block in the main loop.
    json.loads(content) 
    
    return content

def main():
    print(f"Loading dataset from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find the dataset at {CSV_PATH}")
        return

    # 1. Drop rows where 'transcription' is empty
    initial_count = len(df)
    df = df.dropna(subset=['transcription'])
    print(f"Dropped {initial_count - len(df)} rows with missing transcriptions.")

    # 2. Targeted Filtering for spine-related keywords
    keywords = ["sciatica", "lumbar", "radiculopathy", "herniated disc", "spine"]
    
    # Create a regex pattern: 'sciatica|lumbar|radiculopathy|herniated disc|spine'
    pattern = '|'.join(keywords)
    
    # Filter checking either 'description' or 'transcription'
    mask = (
        df['description'].fillna('').str.contains(pattern, case=False, regex=True) |
        df['transcription'].fillna('').str.contains(pattern, case=False, regex=True)
    )
    
    filtered_df = df[mask].copy().head(30)
    print(f"Filtered down to {len(filtered_df)} spine-related records for quick demonstration.")

    if len(filtered_df) == 0:
        print("No records found matching the criteria. Exiting.")
        return

    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)

    print(f"Preparing to extract data. Output will be saved to: {JSONL_PATH}")
    
    # Clear the destination file if it exists, or create it empty
    open(JSONL_PATH, 'w').close()

    success_count = 0
    error_count = 0

    # 3 & 4. AI Synthesizer and Format to JSONL
    # Process each record with a progress bar using tqdm
    for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Processing Records"):
        transcription = row['transcription']
        
        try:
            # Send to Gemini
            extracted_json_str = extract_clinical_info(transcription)
            
            # Format to strict JSONL structure required for LLM fine-tuning
            jsonl_record = {
                "messages": [
                    {"role": "user", "content": transcription},
                    {"role": "model", "content": extracted_json_str}
                ]
            }
            
            # Append line to JSONL file
            with open(JSONL_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(jsonl_record) + '\n')
                
            success_count += 1
            
        except Exception as e:
            # 5. Error Handling: Catch rate limits, API issues, or bad JSON responses
            print(f"\nError processing index {index}: {e}")
            error_count += 1
            
        # Small delay to respect rate limits
        time.sleep(2)

    print("\nProcessing Complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Finetuning data saved to {JSONL_PATH}")

if __name__ == "__main__":
    main()
