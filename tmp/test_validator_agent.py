import asyncio
import sys
import os

# Add the project root to sys.path to allow importing from python_agents
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from core.validator_agent import validate_prognosis
from core.schemas import ClinicalExtraction, Demographics, ValidationResult

async def main():
    print("Testing Validator Agent...")
    clinical_data = ClinicalExtraction(
        demographics=Demographics(age=45, gender="Male", weight="85kg"),
        symptoms=["lower back pain", "shooting pain in left leg"],
        history="3 weeks of persistent pain.",
        suggestedProcedure="Physical Therapy"
    )
    vision_finding = "L4-L5 Disc Herniation"
    
    try:
        result = await validate_prognosis(clinical_data, vision_finding)
        print("\n--- Validator Result ---")
        print(f"Is Safe: {result.isSafe}")
        print(f"Confidence: {result.confidenceScore}%")
        print(f"Risks: {result.risks}")
        print(f"Recommendation: {result.recommendation}")
        print("------------------------\n")
        
        if isinstance(result, ValidationResult):
            print("✅ Validator Agent test passed (Type check)")
        else:
            print("❌ Validator Agent test failed (Type check)")
    except Exception as e:
        print(f"❌ Validator Agent test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
