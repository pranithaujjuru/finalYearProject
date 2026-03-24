import { z } from 'zod';

// Phase 3: Data Flow & Pydantic (Zod) Validation

export const VisionFindingSchema = z.object({
  finding: z.string().describe("The primary finding from the MRI image (e.g., 'L4-L5 herniation')"),
  severity: z.enum(['mild', 'moderate', 'severe']).optional(),
  location: z.string().optional(),
});

export type VisionFinding = z.infer<typeof VisionFindingSchema>;

export const ClinicalExtractionSchema = z.object({
  demographics: z.object({
    age: z.number().optional(),
    gender: z.string().optional(),
    weight: z.string().optional()
  }).describe("Extracted patient demographic information"),
  symptoms: z.array(z.string()).describe("List of symptoms extracted from the report"),
  history: z.string().describe("Patient medical history relevant to sciatica"),
  suggestedProcedure: z.string().describe("Baseline procedure suggested by the NLP agent"),
});

export type ClinicalExtraction = z.infer<typeof ClinicalExtractionSchema>;

export const ValidationResultSchema = z.object({
  isSafe: z.boolean().describe("Whether the suggested procedure is safe based on guidelines"),
  confidenceScore: z.number().min(0).max(100).describe("Confidence in this assessment (0-100)"),
  risks: z.array(z.string()).describe("List of identified safety risks or contraindications"),
  recommendation: z.string().describe("Final recommendation for the clinician"),
  nextSteps: z.array(z.string()).describe("Checklist of immediate action items for the provider"),
  referencedGuidelines: z.array(z.string()).describe("URLs of the specific guidelines used to validate this decision"),
});

export type ValidationResult = z.infer<typeof ValidationResultSchema>;

export const FinalReportSchema = z.object({
  visionFindings: VisionFindingSchema,
  clinicalData: ClinicalExtractionSchema,
  validation: ValidationResultSchema,
  timestamp: z.string(),
});

export type FinalReport = z.infer<typeof FinalReportSchema>;
