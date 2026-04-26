import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid,
} from 'recharts'
import { fetchReports } from '../api.js'
import './Reports.css'

const PERIODS = [
  { value: 'daily',   label: 'Günlük'  },
  { value: 'weekly',  label: 'Haftalık' },
  { value: 'monthly', label: 'Aylık'   },
]

const COLORS = {
  helmet: '#ff6b6b',
  vest:   '#ffd93d',
  mask:   '#6bcbff',
  fire:   '#ff8c42',
}

const LABELS = { helmet: 'Baret', vest: 'Yelek', mask: 'Maske', fire: 'Yangın' }

function today() {
  return new Date().toISOString().slice(0, 10)
}

export default function Reports() {
  const [period, setPeriod]   = useState('weekly')
  const [date, setDate]       = useState(today())
  const [data, setData]       = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchReports(period, period === 'daily' ? date : undefined)
      .then(res => setData(res.data || []))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [period, date])

  const totals = data.reduce(
    (acc, row) => {
      acc.helmet += row.helmet || 0
      acc.vest   += row.vest   || 0
      acc.mask   += row.mask   || 0
      acc.fire   += row.fire   || 0
      acc.total  += row.total  || 0
      return acc
    },
    { helmet: 0, vest: 0, mask: 0, fire: 0, total: 0 }
  )

  const labelKey = period === 'daily' ? 'label' : 'label'

  return (
    <div className="reports-page">
      <div className="rp-header">
        <div className="rp-period-tabs">
          {PERIODS.map(p => (
            <button
              key={p.value}
              className={`rp-tab ${period === p.value ? 'rp-tab--active' : ''}`}
              onClick={() => setPeriod(p.value)}
            >
              {p.label}
            </button>
          ))}
        </div>
        {period === 'daily' && (
          <input
            type="date"
            className="rp-date-input"
            value={date}
            max={today()}
            onChange={e => setDate(e.target.value)}
          />
        )}
      </div>

      {/* Özet kartlar */}
      <div className="rp-summary">
        <div className="rp-sum-card">
          <div className="rp-sum-val">{totals.total}</div>
          <div className="rp-sum-label">Toplam</div>
        </div>
        {Object.keys(COLORS).map(key => (
          <div key={key} className="rp-sum-card" style={{ borderColor: COLORS[key] + '44' }}>
            <div className="rp-sum-val" style={{ color: COLORS[key] }}>{totals[key]}</div>
            <div className="rp-sum-label">{LABELS[key]}</div>
          </div>
        ))}
      </div>

      {/* Grafik */}
      <div className="rp-chart-panel">
        {loading ? (
          <div className="rp-loading">Yükleniyor…</div>
        ) : (
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4e" vertical={false} />
              <XAxis
                dataKey={labelKey}
                tick={{ fill: '#7a8aa0', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                angle={period !== 'daily' ? -35 : 0}
                textAnchor={period !== 'daily' ? 'end' : 'middle'}
                interval={period === 'monthly' ? 2 : 0}
              />
              <YAxis
                tick={{ fill: '#7a8aa0', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                allowDecimals={false}
              />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a4e', borderRadius: 6, fontSize: 12 }}
                labelStyle={{ color: '#c0cde0', marginBottom: 4 }}
                cursor={{ fill: 'rgba(255,255,255,0.03)' }}
              />
              <Legend
                wrapperStyle={{ fontSize: 12, color: '#7a8aa0', paddingTop: 12 }}
                formatter={val => LABELS[val] || val}
              />
              {Object.keys(COLORS).map(key => (
                <Bar key={key} dataKey={key} fill={COLORS[key]} stackId="a" radius={key === 'fire' ? [4,4,0,0] : [0,0,0,0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}
