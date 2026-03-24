from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv(".env.local")
from python_agents.nlp_agent import extract_clinical_data
from python_agents.vision_agent import analyze_mri
from python_agents.validator_agent import validate_prognosis
from python_agents.schemas import FinalReport, VisionFinding
import asyncio
from datetime import datetime

app = FastAPI(title="Sciatica Prognosis AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/prognosis", response_model=FinalReport)
async def get_prognosis(
    mri: UploadFile = File(...),
    reportText: str = Form(...)
):
    try:
        # Read image buffer
        image_bytes = await mri.read()
        mime_type = mri.content_type
        
        print(f"Starting Pipeline for {mri.filename}...")

        # 1. Vision Agent
        print("Triggering Vision Agent...")
        vision_findings = await analyze_mri(image_bytes, mime_type)

        # 2. NLP Agent
        print("Triggering NLP Agent...")
        clinical_data = await extract_clinical_data(reportText)

        # 3. Validator Agent
        print("Triggering Validator Agent...")
        validation = await validate_prognosis(clinical_data, vision_findings.finding)

        # 4. Compile Final Report
        final_report = FinalReport(
            visionFindings=vision_findings,
            clinicalData=clinical_data,
            validation=validation,
            timestamp=datetime.now().isoformat()
        )

        print("Pipeline Complete.")
        return final_report

    except Exception as e:
        print(f"Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
