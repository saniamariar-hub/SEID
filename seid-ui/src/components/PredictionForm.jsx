import React, { useState } from 'react'
import axios from 'axios'

const API_BASE = 'http://127.0.0.1:8001'

const CHANNELS = [
  { value: 'email',   label: 'Email' },
  { value: 'sms',     label: 'SMS' },
  { value: 'unknown', label: 'Unknown' },
]

const MODES = [
  { value: 'balanced',    label: 'Balanced',         desc: 'Equal weight on precision & recall' },
  { value: 'high_recall', label: 'High Recall',      desc: 'Catch more threats (may flag benign)' },
  { value: 'low_fp',      label: 'Low False Positive', desc: 'Minimize false alarms' },
]

function SelectField({ label, value, onChange, options }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-semibold uppercase tracking-widest text-gray-400">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="neon-focus bg-gray-800 border border-gray-600 text-gray-100 rounded-lg px-3 py-2.5
                   text-sm transition-all duration-200 cursor-pointer appearance-none
                   hover:border-cyan-600 focus:border-cyan-400"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  )
}

export default function PredictionForm({ onResult, onError, onLoading }) {
  const [text,    setText]    = useState('')
  const [channel, setChannel] = useState('email')
  const [mode,    setMode]    = useState('balanced')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!text.trim()) {
      onError('Please enter a message to analyze.')
      return
    }
    setLoading(true)
    onLoading(true)
    onError(null)
    try {
      const res = await axios.post(`${API_BASE}/predict`, { text: text.trim(), channel, mode }, { timeout: 30_000 })
      onResult(res.data)
    } catch (err) {
      const msg = err.response?.data?.detail ?? err.message ?? 'Unable to reach the API server.'
      onError(`Analysis failed: ${msg}`)
      onResult(null)
    } finally {
      setLoading(false)
      onLoading(false)
    }
  }

  const charCount = text.length
  const atLimit   = charCount >= 5000

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-5">
      {/* Textarea */}
      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between items-baseline">
          <label className="text-xs font-semibold uppercase tracking-widest text-gray-400">
            Message Content
          </label>
          <span className={`text-[11px] font-mono ${atLimit ? 'text-red-400' : 'text-gray-600'}`}>
            {charCount} / 5000
          </span>
        </div>
        <div className="relative">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, 5000))}
            placeholder="Paste or type the email / SMS message to analyze…"
            rows={7}
            className="neon-focus w-full bg-gray-800 border border-gray-600 text-gray-100 rounded-xl
                       px-4 py-3 text-sm resize-none transition-all duration-200 scrollbar-thin
                       placeholder:text-gray-600 hover:border-cyan-700 focus:border-cyan-400"
          />
        </div>
      </div>

      {/* Controls row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <SelectField label="Channel" value={channel} onChange={setChannel} options={CHANNELS} />
        <SelectField label="Detection Mode" value={mode} onChange={setMode} options={MODES} />
      </div>

      {/* Mode hint */}
      <p className="text-[11px] text-gray-500 -mt-2 font-mono">
        {MODES.find(m => m.value === mode)?.desc}
      </p>

      {/* Submit */}
      <button
        type="submit"
        disabled={loading}
        className={`relative flex items-center justify-center gap-2 rounded-xl px-6 py-3.5 text-sm font-bold
                    uppercase tracking-widest transition-all duration-200 border
                    ${loading
                      ? 'bg-gray-800 border-gray-600 text-gray-500 cursor-not-allowed'
                      : 'bg-cyan-500/10 border-cyan-500 text-cyan-400 hover:bg-cyan-500/20 hover:shadow-[0_0_16px_rgba(34,211,238,0.3)] active:scale-[0.98]'
                    }`}
      >
        {loading ? (
          <>
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
            </svg>
            Analyzing message…
          </>
        ) : (
          'Run Analysis'
        )}
      </button>
    </form>
  )
}

