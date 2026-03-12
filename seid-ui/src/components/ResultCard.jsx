import React from 'react'

const TIER_CONFIG = {
  Low:      { bar: 'bg-green-400',  glow: 'shadow-[0_0_12px_rgba(74,222,128,0.4)]',  border: 'border-green-500/50',  text: 'text-green-400',  bg: 'bg-green-500/10' },
  Medium:   { bar: 'bg-yellow-400', glow: 'shadow-[0_0_12px_rgba(250,204,21,0.4)]',  border: 'border-yellow-500/50', text: 'text-yellow-400', bg: 'bg-yellow-500/10' },
  High:     { bar: 'bg-orange-400', glow: 'shadow-[0_0_12px_rgba(251,146,60,0.4)]',  border: 'border-orange-500/50', text: 'text-orange-400', bg: 'bg-orange-500/10' },
  Critical: { bar: 'bg-red-500',    glow: 'shadow-[0_0_16px_rgba(239,68,68,0.55)]',  border: 'border-red-500/60',    text: 'text-red-400',    bg: 'bg-red-500/10'  },
}

function MetaBadge({ label, value, mono = false }) {
  return (
    <div className="flex flex-col gap-0.5 bg-gray-800/60 border border-gray-700/70 rounded-lg px-3 py-2">
      <span className="text-[10px] uppercase tracking-widest text-gray-500">{label}</span>
      <span className={`text-sm font-semibold text-gray-200 ${mono ? 'font-mono' : ''}`}>{value}</span>
    </div>
  )
}

export default function ResultCard({ result }) {
  if (!result) return null

  const { probability, risk_tier, is_malicious, channel, mode, roberta_enabled } = result
  const tier   = TIER_CONFIG[risk_tier] ?? TIER_CONFIG['Medium']
  const pct    = Math.round(probability * 100)
  const label  = risk_tier ?? 'Unknown'

  return (
    <div className={`relative rounded-2xl border ${tier.border} ${tier.glow} bg-gray-900/80 p-6 backdrop-blur-sm overflow-hidden scanline`}>

      {/* Corner decoration */}
      <div className={`absolute top-0 right-0 w-24 h-24 rounded-bl-full opacity-10 ${tier.bg}`} />

      {/* Header: risk tier scale */}
      <div className="mb-5">
        <span className="text-xs uppercase tracking-widest text-gray-500 mb-2 block">Risk Tier</span>
        <div className="flex gap-2">
          {['Low', 'Medium', 'High', 'Critical'].map((t) => {
            const cfg = TIER_CONFIG[t]
            const active = t === label
            return (
              <div
                key={t}
                className={`flex-1 text-center py-1.5 rounded-lg border text-xs font-bold tracking-wide transition-all duration-300 ${
                  active
                    ? `${cfg.bg} ${cfg.border} ${cfg.text} ${cfg.glow}`
                    : 'bg-gray-800/40 border-gray-700/40 text-gray-600'
                }`}
              >
                {t}
              </div>
            )
          })}
        </div>
      </div>

      {/* Probability bar */}
      <div className="mb-5">
        <div className="flex justify-between text-xs mb-1.5">
          <span className="text-gray-400 uppercase tracking-widest">Threat Probability</span>
          <span className={`font-mono font-bold text-base ${tier.text}`}>{pct}%</span>
        </div>
        <div className="w-full bg-gray-800 rounded-full h-3 overflow-hidden border border-gray-700">
          <div
            className={`h-full rounded-full transition-all duration-700 ease-out ${tier.bar}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-gray-700 mt-1 font-mono">
          <span>0% – Safe</span>
          <span>100% – Critical</span>
        </div>
      </div>

      {/* Meta badges */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <MetaBadge label="Channel"  value={channel?.toUpperCase() ?? '—'} mono />
        <MetaBadge label="Mode"     value={mode ?? '—'} />
        <MetaBadge label="Score"    value={`${probability?.toFixed(4)}`} mono />
        <MetaBadge label="RoBERTa"  value={roberta_enabled ? 'Active' : 'Off'} />
      </div>
    </div>
  )
}

