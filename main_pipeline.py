import asyncio
import json
import os
from dotenv import load_dotenv
load_dotenv(".env.local")
from python_agents.nlp_agent import extract_clinical_data, process_patient_intake
from python_agents.vision_agent import analyze_mri
from python_agents.validator_agent import validate_prognosis
from python_agents.schemas import PatientQuestionnaire

async def get_vision_findings():
    # Mocking the vision finding retrieval similar to main_pipeline.ts
    # In a real scenario, this might read from a file or wait for Agent 1
    bridge_path = os.path.join(os.path.dirname(__file__), 'bridge', 'vision_output.json')
    try:
        with open(bridge_path, 'r') as f:
            data = json.load(f)
            return data.get('finding', "No finding string provided in vision output.")
    except FileNotFoundError:
        print("⚠️ Vision output file not found. Defaulting to a mock finding for demonstration.")
        return "Severe L4-L5 Disc Herniation"

async def run_clinical_pipeline(input_data, source_type: str):
    print(f"🚀 Starting Clinical Prognosis Pipeline (Source: {source_type})")
    print("=" * 50)
    
    # Step A: Get Vision Findings
    print("📸 [Agent 1] Retrieving Vision (MRI) Findings...")
    vision_data = await get_vision_findings()
    print(f"   Result: \"{vision_data}\"\n")
    
    # Step B: Extract Clinical Data via NLP Agent
    print("📝 [Agent 2] Extracting Clinical Data via NLP Agent...")
    if source_type == "clinical_upload":
        patient_text_for_report = input_data
        nlp_data = await extract_clinical_data(patient_text_for_report)
    else:
        patient_text_for_report = json.dumps(input_data.dict(), indent=2)
        nlp_data = await process_patient_intake(input_data)
    
    print(f"   Symptoms: {', '.join(nlp_data.symptoms)}")
    print(f"   Procedure: {nlp_data.suggestedProcedure}\n")
    
    # Step C: Validate Prognosis via Validator Agent
    print("🔬 [Agent 3] Validating Prognosis & Checking Guidelines...")
    validation_result = await validate_prognosis(nlp_data, vision_data)
    print(f"   Safe? {'✅ Yes' if validation_result.isSafe else '❌ No'}")
    print(f"   Recommendation: {validation_result.recommendation}\n")
    
    # Step 4: Generate Report
    print("📄 Generating Final Report...")
    report = f"""# Clinical Prognosis Report

## Patient History
**Source:** {source_type}
**Input:** 
> {patient_text_for_report}

**Demographics:**
- Age: {nlp_data.demographics.age or 'N/A'}
- Gender: {nlp_data.demographics.gender or 'N/A'}
- Weight: {nlp_data.demographics.weight or 'N/A'}

**Extracted Symptoms:** 
{chr(10).join([f"- {s}" for s in nlp_data.symptoms])}

**History Details:** 
{nlp_data.history}

## Radiology Findings (Vision Agent)
- {vision_data}

## Proposed Intervention (NLP Agent)
- **Baseline Plan:** {nlp_data.suggestedProcedure}

## Safety & Guidelines Validation (Validator Agent)
- **Safe to Proceed:** {'YES' if validation_result.isSafe else 'NO'}
- **Confidence Score:** {validation_result.confidenceScore}%
- **Final Recommendation:** {validation_result.recommendation}

**Identified Risks / Red Flags:**
{chr(10).join([f"- {r}" for r in validation_result.risks]) if validation_result.risks else "None identified."}

**Immediate Next Steps:**
{chr(10).join([f"- [ ] {step}" for step in validation_result.nextSteps]) if hasattr(validation_result, 'nextSteps') and validation_result.nextSteps else "No immediate steps provided."}

**Referenced Medical Guidelines:**
{chr(10).join([f"- [{g}]({g})" for g in validation_result.referencedGuidelines]) if validation_result.referencedGuidelines else "No specific external guidelines referenced."}
"""

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, f"patient_prognosis_report_{source_type}.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Final Report saved to: {report_path}")

async def run_tests():
    mock_clinical_text = "A 50-year-old patient presents with shooting pain down the back of their left leg that worsens when sitting. They also report numbness in their lower leg and mild weakness when trying to lift their foot."
    
    mock_questionnaire = PatientQuestionnaire(
        painTrajectory="shooting pain down the back of the left leg",
        painDuration="3 weeks",
        previousTreatments="ibuprofen, rest",
        redFlags="numbness in lower leg, mild weakness lifting foot",
        neurologicalSymptoms="weakness when lifting foot, numbness"
    )

    print("=== TEST RUN 1: CLINICAL UPLOAD ===")
    await run_clinical_pipeline(mock_clinical_text, "clinical_upload")

    print("\n=== TEST RUN 2: PATIENT QUESTIONNAIRE ===")
    await run_clinical_pipeline(mock_questionnaire, "patient_questionnaire")

if __name__ == "__main__":
    asyncio.run(run_tests())
