import asyncio
import sys
import os

# Add the project root to sys.path to allow importing from python_agents
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from core.nlp_agent import extract_clinical_data
from core.schemas import ClinicalExtraction

async def main():
    print("Testing NLP Agent...")
    report_text = "Patient is a 45-year-old male, weight 85kg. Reports lower back pain radiating to left leg for 3 weeks. History of mild hypertension. Suggested procedure: Physical Therapy."
    try:
        result = await extract_clinical_data(report_text)
        print("\n--- NLP Result ---")
        print(f"Demographics: {result.demographics}")
        print(f"Symptoms: {result.symptoms}")
        print(f"History: {result.history}")
        print(f"Suggested Procedure: {result.suggestedProcedure}")
        print("------------------\n")
        
        if isinstance(result, ClinicalExtraction):
            print("✅ NLP Agent test passed (Type check)")
        else:
            print("❌ NLP Agent test failed (Type check)")
    except Exception as e:
        print(f"❌ NLP Agent test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
