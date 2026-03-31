import asyncio
import sys
import os

# Add the project root to sys.path to allow importing from python_agents
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from core.translator_agent import translate_to_patient_voice
from core.schemas import FinalReport, VisionFinding, ClinicalExtraction, Demographics, ValidationResult, PatientVisitSummary

async def main():
    print("Testing Translator Agent...")
    report = FinalReport(
        visionFindings=VisionFinding(finding="L4-L5 Disc Herniation", severity="moderate", location="L4-L5"),
        clinicalData=ClinicalExtraction(
            demographics=Demographics(age=45, gender="Male", weight="85kg"),
            symptoms=["lower back pain", "shooting pain in left leg"],
            history="3 weeks of pain.",
            suggestedProcedure="Physical Therapy"
        ),
        validation=ValidationResult(
            isSafe=True,
            confidenceScore=85.0,
            risks=["None"],
            recommendation="Proceed with physical therapy.",
            nextSteps=["Schedule PT session"],
            referencedGuidelines=["https://example.com/guidelines"]
        )
    )
    
    try:
        result = await translate_to_patient_voice(report)
        print("\n--- Translator Result ---")
        print(f"Title: {result.summaryTitle}")
        print(f"Diagnosis: {result.diagnosis}")
        print(f"Plan: {result.plan}")
        print("-------------------------\n")
        
        if isinstance(result, PatientVisitSummary):
            print("✅ Translator Agent test passed (Type check)")
        else:
            print("❌ Translator Agent test failed (Type check)")
    except Exception as e:
        print(f"❌ Translator Agent test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
