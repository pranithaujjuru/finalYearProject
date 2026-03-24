import React, { useState } from 'react';
import { 
  Activity, 
  FileText, 
  Image as ImageIcon, 
  ShieldCheck, 
  AlertCircle, 
  ChevronRight, 
  Loader2,
  CheckCircle2,
  Stethoscope
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import type { FinalReport } from './types/schemas';

export default function App() {
  const [mriFile, setMriFile] = useState<File | null>(null);
  const [reportText, setReportText] = useState('');
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<FinalReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState<number>(0);

  const steps = [
    { id: 1, name: 'Vision Agent', description: 'Analyzing MRI scan...' },
    { id: 2, name: 'NLP Agent', description: 'Extracting clinical data...' },
    { id: 3, name: 'Validator Agent', description: 'Safety check & guidelines...' },
    { id: 4, name: 'Final Report', description: 'Compiling results...' }
  ];

  const handleRunPrognosis = async () => {
    if (!mriFile || !reportText) {
      setError('Please provide both an MRI image and clinical report text.');
      return;
    }

    setLoading(true);
    setError(null);
    setReport(null);
    setStep(0);

    const formData = new FormData();
    formData.append('mri', mriFile);
    formData.append('reportText', reportText);

    try {
      // Simulate step progress
      const interval = setInterval(() => {
        setStep(prev => (prev < 3 ? prev + 1 : prev));
      }, 2000);

      const response = await fetch('/api/prognosis', {
        method: 'POST',
        body: formData,
      });

      clearInterval(interval);

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Failed to run prognosis');
      }

      const data = await response.json();
      setReport(data);
      setStep(4);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-4 md:p-8 max-w-6xl mx-auto">
      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-600 rounded-lg text-white">
            <Stethoscope size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Sciatica Prognosis AI</h1>
            <p className="text-slate-500 text-sm">Multi-Agent Diagnostic Support System</p>
          </div>
        </div>
        <div className="hidden md:block px-3 py-1 bg-slate-100 rounded-full text-xs font-medium text-slate-600">
          v1.0.0-beta
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Input Section */}
        <div className="lg:col-span-5 space-y-6">
          <section className="medical-card p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <ImageIcon size={20} className="text-blue-500" />
              MRI Scan Upload
            </h2>
            <div 
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${
                mriFile ? 'border-blue-400 bg-blue-50/50' : 'border-slate-200 hover:border-blue-300'
              }`}
              onClick={() => document.getElementById('mri-upload')?.click()}
            >
              <input 
                id="mri-upload"
                type="file" 
                className="hidden" 
                accept="image/*"
                onChange={(e) => setMriFile(e.target.files?.[0] || null)}
              />
              {mriFile ? (
                <div className="flex flex-col items-center">
                  <CheckCircle2 className="text-blue-500 mb-2" size={32} />
                  <p className="text-sm font-medium text-slate-700">{mriFile.name}</p>
                  <p className="text-xs text-slate-500 mt-1">Click to change</p>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  <ImageIcon className="text-slate-300 mb-2" size={32} />
                  <p className="text-sm text-slate-600">Drag & drop or click to upload MRI</p>
                  <p className="text-xs text-slate-400 mt-1">Supports JPG, PNG, DICOM (as image)</p>
                </div>
              )}
            </div>
          </section>

          <section className="medical-card p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <FileText size={20} className="text-blue-500" />
              Clinical Report
            </h2>
            <textarea 
              className="w-full h-48 p-4 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all text-sm"
              placeholder="Paste clinical report text here... (e.g., Patient reports radiating pain in left leg, numbness in toes...)"
              value={reportText}
              onChange={(e) => setReportText(e.target.value)}
            />
          </section>

          <button 
            disabled={loading}
            onClick={handleRunPrognosis}
            className={`w-full py-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all ${
              loading 
                ? 'bg-slate-100 text-slate-400 cursor-not-allowed' 
                : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-200 active:scale-[0.98]'
            }`}
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin" size={20} />
                Processing Agents...
              </>
            ) : (
              <>
                <Activity size={20} />
                Run AI Prognosis
              </>
            )}
          </button>

          {error && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 bg-red-50 border border-red-100 rounded-xl flex items-start gap-3 text-red-700 text-sm"
            >
              <AlertCircle size={18} className="shrink-0 mt-0.5" />
              <p>{error}</p>
            </motion.div>
          )}
        </div>

        {/* Output Section */}
        <div className="lg:col-span-7">
          <AnimatePresence mode="wait">
            {!loading && !report ? (
              <motion.div 
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-full flex flex-col items-center justify-center text-center p-12 border border-slate-100 rounded-2xl bg-slate-50/50"
              >
                <div className="w-16 h-16 bg-white rounded-2xl shadow-sm flex items-center justify-center mb-4">
                  <Activity size={32} className="text-slate-300" />
                </div>
                <h3 className="text-lg font-medium text-slate-900">Ready for Analysis</h3>
                <p className="text-slate-500 text-sm max-w-xs mt-2">
                  Upload patient data to trigger the 4-agent prognosis pipeline.
                </p>
              </motion.div>
            ) : loading ? (
              <motion.div 
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-6"
              >
                <div className="medical-card p-8 text-center">
                  <div className="relative w-24 h-24 mx-auto mb-6">
                    <div className="absolute inset-0 border-4 border-blue-100 rounded-full"></div>
                    <div className="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Activity size={32} className="text-blue-600" />
                    </div>
                  </div>
                  <h3 className="text-xl font-bold mb-2">Orchestrating Agents</h3>
                  <p className="text-slate-500">The system is coordinating multiple AI models to analyze the case.</p>
                </div>

                <div className="space-y-3">
                  {steps.map((s, i) => (
                    <div 
                      key={s.id}
                      className={`p-4 rounded-xl border transition-all flex items-center justify-between ${
                        step >= i 
                          ? 'bg-white border-blue-100 shadow-sm' 
                          : 'bg-slate-50 border-slate-100 opacity-50'
                      }`}
                    >
                      <div className="flex items-center gap-4">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                          step > i ? 'bg-green-500 text-white' : step === i ? 'bg-blue-600 text-white' : 'bg-slate-200 text-slate-500'
                        }`}>
                          {step > i ? <CheckCircle2 size={16} /> : s.id}
                        </div>
                        <div>
                          <p className="font-semibold text-sm">{s.name}</p>
                          <p className="text-xs text-slate-500">{s.description}</p>
                        </div>
                      </div>
                      {step === i && <Loader2 className="animate-spin text-blue-600" size={16} />}
                    </div>
                  ))}
                </div>
              </motion.div>
            ) : report && (
              <motion.div 
                key="report"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-6"
              >
                {/* Vision Findings */}
                <section className="medical-card overflow-hidden">
                  <div className="bg-slate-900 p-4 flex items-center justify-between">
                    <h3 className="text-white font-medium flex items-center gap-2">
                      <ImageIcon size={18} className="text-blue-400" />
                      Vision Agent Analysis
                    </h3>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                      report.visionFindings.severity === 'severe' ? 'bg-red-500 text-white' :
                      report.visionFindings.severity === 'moderate' ? 'bg-orange-500 text-white' : 'bg-green-500 text-white'
                    }`}>
                      {report.visionFindings.severity}
                    </span>
                  </div>
                  <div className="p-6">
                    <p className="text-2xl font-bold text-slate-900 mb-1">{report.visionFindings.finding}</p>
                    <p className="text-slate-500 text-sm flex items-center gap-1">
                      <ChevronRight size={14} /> Location: {report.visionFindings.location}
                    </p>
                  </div>
                </section>

                {/* Clinical Data */}
                <section className="medical-card p-6">
                  <h3 className="text-slate-900 font-semibold mb-4 flex items-center gap-2">
                    <FileText size={18} className="text-blue-500" />
                    Clinical Extraction (NLP)
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Symptoms</h4>
                      <div className="flex flex-wrap gap-2">
                        {report.clinicalData.symptoms.map((s, i) => (
                          <span key={i} className="px-2 py-1 bg-slate-100 text-slate-700 rounded text-xs font-medium">
                            {s}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Suggested Procedure</h4>
                      <p className="text-sm text-slate-700 font-medium">{report.clinicalData.suggestedProcedure}</p>
                    </div>
                  </div>
                  <div className="mt-6 pt-6 border-t border-slate-100">
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Medical History</h4>
                    <p className="text-sm text-slate-600 leading-relaxed">{report.clinicalData.history}</p>
                  </div>
                </section>

                {/* Validation & Safety */}
                <section className={`medical-card p-6 border-l-4 ${report.validation.isSafe ? 'border-l-green-500' : 'border-l-red-500'}`}>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-slate-900 font-semibold flex items-center gap-2">
                      <ShieldCheck size={18} className={report.validation.isSafe ? 'text-green-500' : 'text-red-500'} />
                      Safety Validation
                    </h3>
                    <div className={`px-3 py-1 rounded-full text-xs font-bold flex items-center gap-1.5 ${
                      report.validation.isSafe ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                    }`}>
                      {report.validation.isSafe ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
                      {report.validation.isSafe ? 'SAFE' : 'RISK DETECTED'}
                    </div>
                  </div>
                  
                  {report.validation.risks.length > 0 && (
                    <div className="mb-4 space-y-2">
                      {report.validation.risks.map((risk, i) => (
                        <div key={i} className="flex items-start gap-2 text-sm text-slate-700 bg-slate-50 p-2 rounded-lg">
                          <AlertCircle size={14} className="text-red-500 shrink-0 mt-1" />
                          {risk}
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="bg-slate-900 text-white p-4 rounded-xl">
                    <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Final Recommendation</h4>
                    <p className="text-sm font-medium">{report.validation.recommendation}</p>
                  </div>
                </section>

                <div className="text-center text-[10px] text-slate-400">
                  Report generated on {new Date(report.timestamp).toLocaleString()} • AI-generated prognosis for clinical support only.
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
