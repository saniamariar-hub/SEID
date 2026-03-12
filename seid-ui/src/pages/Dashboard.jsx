import React, { useState } from 'react'
import HealthStatus   from '../components/HealthStatus.jsx'
import PredictionForm from '../components/PredictionForm.jsx'
import ResultCard     from '../components/ResultCard.jsx'

function ShieldIcon() {
  return (
    <svg className="w-8 h-8 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6
           2.25 7.235v.658c0 5.607 3.694 10.375 8.652 12.103a.75.75
           0 00.696 0C17.556 17.368 21.25 12.6 21.25 6.993v-.658a11.96
           11.96 0 00-6.348-1.778H12z"
      />
    </svg>
  )
}

function GridBackground() {
  return (
    <div
      className="pointer-events-none fixed inset-0 z-0"
      style={{
        backgroundImage: `
          linear-gradient(rgba(34,211,238,0.04) 1px, transparent 1px),
          linear-gradient(90deg, rgba(34,211,238,0.04) 1px, transparent 1px)
        `,
        backgroundSize: '40px 40px',
      }}
    />
  )
}

export default function Dashboard() {
  const [result,  setResult]  = useState(null)
  const [error,   setError]   = useState(null)
  const [loading, setLoading] = useState(false)

  return (
    <div className="relative min-h-screen bg-gray-950 text-gray-100 overflow-x-hidden">
      <GridBackground />

      {/* Glow orb top-left */}
      <div className="pointer-events-none fixed -top-40 -left-40 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
      {/* Glow orb bottom-right */}
      <div className="pointer-events-none fixed -bottom-40 -right-40 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl" />

      <div className="relative z-10 max-w-3xl mx-auto px-4 py-10 sm:py-16">

        {/* ── Header ── */}
        <header className="mb-8">
          <div className="flex items-center gap-3 mb-3">
            <ShieldIcon />
            <span className="text-xs font-mono uppercase tracking-[0.3em] text-cyan-600">v3.1.0</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-white leading-tight tracking-tight">
            SEID Engine
            <span className="text-cyan-400"> –</span> Social Engineering
            <br className="hidden sm:block" /> Detection System
          </h1>
          <p className="mt-3 text-gray-400 text-sm sm:text-base max-w-xl">
            Analyze messages for phishing and smishing attacks using an ensemble of
            <span className="text-cyan-400"> TF-IDF + Logistic Regression</span> and
            <span className="text-cyan-400"> RoBERTa</span> transformer models.
          </p>
        </header>

        {/* ── Health Status ── */}
        <div className="mb-6">
          <HealthStatus />
        </div>

        {/* ── Main card ── */}
        <div className="rounded-2xl border border-gray-700/60 bg-gray-900/60 backdrop-blur-sm p-6 sm:p-8 shadow-2xl">
          <div className="flex items-center gap-2 mb-6 pb-4 border-b border-gray-800">
            <h2 className="text-sm font-bold uppercase tracking-widest text-gray-300">Threat Analysis</h2>
          </div>

          <PredictionForm
            onResult={setResult}
            onError={setError}
            onLoading={setLoading}
          />
        </div>

        {/* ── Error Alert ── */}
        {error && (
          <div className="mt-5 flex items-start gap-3 rounded-xl border border-red-500/40 bg-red-500/10 px-5 py-4">
            <span className="text-red-400 mt-0.5 text-sm font-bold flex-shrink-0">!</span>
            <p className="text-red-300 text-sm">{error}</p>
            <button onClick={() => setError(null)} className="ml-auto text-red-500 hover:text-red-300 text-lg leading-none flex-shrink-0">×</button>
          </div>
        )}

        {/* ── Analyzing overlay hint ── */}
        {loading && !error && (
          <div className="mt-5 flex items-center gap-3 rounded-xl border border-cyan-500/30 bg-cyan-500/5 px-5 py-4">
            <svg className="w-4 h-4 animate-spin text-cyan-400 flex-shrink-0" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
            </svg>
            <p className="text-cyan-300 text-sm font-mono">Running ensemble model inference…</p>
          </div>
        )}

        {/* ── Result Card ── */}
        {result && !loading && (
          <div className="mt-5">
            <div className="flex items-center gap-2 mb-3">
              <h2 className="text-xs font-bold uppercase tracking-widest text-gray-400">Analysis Result</h2>
            </div>
            <ResultCard result={result} />
          </div>
        )}

        {/* ── Footer ── */}
        <footer className="mt-14 text-center text-[11px] font-mono text-gray-700 space-y-1">
          <p>SEID Engine · Ensemble ML Threat Detection</p>
          <p className="text-gray-800">API -- http://127.0.0.1:8001</p>
        </footer>
      </div>
    </div>
  )
}

