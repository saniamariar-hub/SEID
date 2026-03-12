import React, { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const API_BASE = 'http://127.0.0.1:8001'

function Dot({ color }) {
  const colors = {
    green: 'bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.8)]',
    red:   'bg-red-500  shadow-[0_0_8px_rgba(239,68,68,0.8)]',
    gray:  'bg-gray-500',
  }
  return (
    <span className={`inline-block w-2.5 h-2.5 rounded-full animate-pulse ${colors[color]}`} />
  )
}

function Badge({ label, value, mono = false }) {
  return (
    <div className="flex items-center gap-2 bg-gray-800/60 border border-gray-700 rounded-lg px-3 py-1.5">
      <span className="text-gray-400 text-xs uppercase tracking-widest">{label}</span>
      <span className={`text-cyan-300 text-xs font-semibold ${mono ? 'font-mono' : ''}`}>{value}</span>
    </div>
  )
}

export default function HealthStatus() {
  const [health, setHealth]   = useState(null)
  const [status, setStatus]   = useState('loading') // 'loading' | 'ok' | 'error'
  const [lastChecked, setLastChecked] = useState(null)

  const fetchHealth = useCallback(async () => {
    setStatus('loading')
    try {
      const res = await axios.get(`${API_BASE}/health`, { timeout: 5000 })
      setHealth(res.data)
      setStatus('ok')
    } catch {
      setHealth(null)
      setStatus('error')
    } finally {
      setLastChecked(new Date().toLocaleTimeString())
    }
  }, [])

  useEffect(() => {
    fetchHealth()
    const id = setInterval(fetchHealth, 30_000)
    return () => clearInterval(id)
  }, [fetchHealth])

  return (
    <div className="relative flex flex-wrap items-center gap-3 rounded-xl border border-gray-700/60 bg-gray-900/70 px-5 py-3 backdrop-blur-sm">
      {/* Status dot + label */}
      <div className="flex items-center gap-2 mr-1">
        {status === 'loading' && <Dot color="gray" />}
        {status === 'ok'      && <Dot color="green" />}
        {status === 'error'   && <Dot color="red" />}
        <span className="text-xs font-semibold uppercase tracking-widest text-gray-300">
          {status === 'loading' ? 'Connecting…'
           : status === 'ok'   ? 'API Online'
           : 'API Unreachable'}
        </span>
      </div>

      {status === 'ok' && health && (
        <>
          <Badge label="RoBERTa" value={health.roberta_enabled ? 'Enabled' : 'Disabled'} />
          <Badge label="Device"  value={health.device?.toUpperCase() ?? '—'} mono />
        </>
      )}

      {/* Last checked + refresh */}
      <div className="ml-auto flex items-center gap-2">
        {lastChecked && (
          <span className="text-gray-600 text-[11px] font-mono">
            checked {lastChecked}
          </span>
        )}
        <button
          onClick={fetchHealth}
          title="Refresh health"
          className="text-gray-500 hover:text-cyan-400 transition-colors text-xs px-1.5 py-0.5 rounded border border-gray-700 hover:border-cyan-600"
        >
          ↻
        </button>
      </div>
    </div>
  )
}

