from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class VisionFinding(BaseModel):
    finding: str = Field(description="The primary finding from the MRI image (e.g., 'L4-L5 herniation')")
    severity: Optional[str] = Field(None, description="Severity: mild, moderate, or severe")
    location: Optional[str] = Field(None, description="Anatomical location of the finding")

class Demographics(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    weight: Optional[str] = None

class ClinicalExtraction(BaseModel):
    demographics: Demographics = Field(description="Extracted patient demographic information")
    symptoms: List[str] = Field(description="List of symptoms extracted from the report")
    history: str = Field(description="Patient medical history relevant to sciatica")
    suggestedProcedure: str = Field(description="Baseline procedure suggested by the NLP agent")

class ValidationResult(BaseModel):
    isSafe: bool = Field(description="Whether the suggested procedure is safe based on guidelines")
    confidenceScore: float = Field(description="Confidence in this assessment (0-100)")
    risks: List[str] = Field(description="List of identified safety risks or contraindications")
    recommendation: str = Field(description="Final recommendation for the clinician")
    nextSteps: List[str] = Field(description="Checklist of immediate action items for the provider")
    referencedGuidelines: List[str] = Field(description="URLs of the specific guidelines used to validate this decision")

class PatientVisitSummary(BaseModel):
    summaryTitle: str = Field(description="A concise, empathetic title for the patient's visit.")
    diagnosis: str = Field(description="Empathetic explanation of the suspected condition with analogies.")
    neurological: str = Field(description="Explanation of nerve/muscle findings in simple terms.")
    imaging: str = Field(description="Plain-English translation of MRI findings.")
    plan: str = Field(description="The suggested care plan described constructively.")
    redFlags: str = Field(description="Safety warnings explained calmly but clearly.")
    jargonBuster: List[dict] = Field(description="List of dicts with {'term': '...', 'explanation': '...'} for complex terms.")

class FinalReport(BaseModel):
    visionFindings: VisionFinding
    clinicalData: ClinicalExtraction
    validation: ValidationResult
    patientSummary: Optional[PatientVisitSummary] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class PatientQuestionnaire(BaseModel):
    painTrajectory: str
    painDuration: str
    previousTreatments: str
    redFlags: str
    neurologicalSymptoms: str
