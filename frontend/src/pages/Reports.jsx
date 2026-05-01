import { useEffect, useState, useRef } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid,
} from 'recharts'
import { fetchReports, fetchReportSummary, generateReportLLM } from '../api.js'
import socket from '../socket.js'
import './Reports.css'

const PERIODS = [
  { value: 'daily',   label: 'Günlük'   },
  { value: 'weekly',  label: 'Haftalık'  },
  { value: 'monthly', label: 'Aylık'    },
]

const COLORS = { helmet: '#ff6b6b', vest: '#ffd93d', mask: '#6bcbff', fire: '#ff8c42' }
const LABELS = { helmet: 'Baret', vest: 'Yelek', mask: 'Maske', fire: 'Yangın' }

const RISK_COLORS = {
  low:      '#4ade80',
  medium:   '#fbbf24',
  high:     '#fb923c',
  critical: '#ef4444',
}
const RISK_LABELS = { low: 'Düşük', medium: 'Orta', high: 'Yüksek', critical: 'Kritik' }

const TREND_META = {
  increasing: { icon: '↑', color: '#ef4444', label: 'Artış'   },
  decreasing: { icon: '↓', color: '#4ade80', label: 'Azalış'  },
  stable:     { icon: '→', color: '#7a8aa0', label: 'Stabil'  },
  no_data:    { icon: '–', color: '#7a8aa0', label: 'Veri yok' },
}

function today() { return new Date().toISOString().slice(0, 10) }

export default function Reports() {
  const [period, setPeriod]       = useState('weekly')
  const [date, setDate]           = useState(today())
  const [chartData, setChartData] = useState([])
  const [summary, setSummary]     = useState(null)
  const [llmText, setLlmText]     = useState('')
  const [chartLoading, setChartLoading]     = useState(true)
  const [summaryLoading, setSummaryLoading] = useState(true)
  const [llmLoading, setLlmLoading]         = useState(false)
  const [llmError, setLlmError]             = useState('')
  const pendingRef = useRef(null)   // { period, date } of in-flight request

  const dateParam = period === 'daily' ? date : undefined

  useEffect(() => {
    setChartLoading(true)
    fetchReports(period, dateParam)
      .then(res => setChartData(res.data || []))
      .catch(console.error)
      .finally(() => setChartLoading(false))
  }, [period, date])

  useEffect(() => {
    setSummaryLoading(true)
    setSummary(null)
    setLlmText('')
    setLlmError('')
    fetchReportSummary(period, dateParam)
      .then(setSummary)
      .catch(() => setSummary(null))
      .finally(() => setSummaryLoading(false))
  }, [period, date])

  // Socket.IO — LLM tamamlandığında
  useEffect(() => {
    function onReady({ period: p, date: d, llm_text }) {
      const pending = pendingRef.current
      if (!pending) return
      if (pending.period === p && pending.date === (d || '')) {
        setLlmText(llm_text || '')
        setLlmLoading(false)
        pendingRef.current = null
      }
    }
    function onError({ period: p, date: d, error }) {
      const pending = pendingRef.current
      if (!pending) return
      if (pending.period === p && pending.date === (d || '')) {
        setLlmError('Rapor oluşturulamadı: ' + error)
        setLlmLoading(false)
        pendingRef.current = null
      }
    }
    socket.on('report_llm_ready', onReady)
    socket.on('report_llm_error', onError)
    return () => {
      socket.off('report_llm_ready', onReady)
      socket.off('report_llm_error', onError)
    }
  }, [])

  const totals = chartData.reduce(
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

  const risk        = summary?.risk_summary || {}
  const comparison  = summary?.comparison   || {}
  const locations   = (summary?.location_breakdown || []).filter(l => l.camera_id !== 'unknown')
  const topLocation = locations[0] || null
  const trend       = TREND_META[comparison.trend] || TREND_META.no_data
  const riskColor   = RISK_COLORS[risk.risk_level] || '#7a8aa0'

  const handleGenerateLLM = () => {
    setLlmLoading(true)
    setLlmText('')
    setLlmError('')
    pendingRef.current = { period, date: dateParam || '' }
    generateReportLLM(period, dateParam)
      .catch(() => {
        setLlmError('Sunucu hatası.')
        setLlmLoading(false)
        pendingRef.current = null
      })
  }

  return (
    <div className="reports-page">

      {/* ── Header ── */}
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

      {/* ── İhlal özet kartlar ── */}
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

      {/* ── Risk / Trend / Lokasyon kartlar ── */}
      <div className="rp-info-row">

        {/* Risk kartı */}
        <div className="rp-info-card">
          <div className="rp-info-label">Risk Seviyesi</div>
          {summaryLoading ? (
            <div className="rp-info-placeholder">…</div>
          ) : (
            <>
              <div className="rp-risk-badge" style={{ color: riskColor, borderColor: riskColor + '44' }}>
                {RISK_LABELS[risk.risk_level] || '—'}
              </div>
              <div className="rp-risk-bar-wrap">
                <div
                  className="rp-risk-bar-fill"
                  style={{ width: `${risk.normalized_score || 0}%`, background: riskColor }}
                />
              </div>
              <div className="rp-risk-score" style={{ color: riskColor }}>
                {risk.normalized_score ?? '—'}<span>/100</span>
              </div>
            </>
          )}
        </div>

        {/* Trend kartı */}
        <div className="rp-info-card">
          <div className="rp-info-label">Önceki Dönemle Karşılaştırma</div>
          {summaryLoading ? (
            <div className="rp-info-placeholder">…</div>
          ) : comparison.trend === 'no_data' ? (
            <div className="rp-trend-nodata">Karşılaştırılacak veri yok</div>
          ) : (
            <>
              <div className="rp-trend-icon" style={{ color: trend.color }}>
                {trend.icon}
              </div>
              <div className="rp-trend-pct" style={{ color: trend.color }}>
                {comparison.change_percent !== null
                  ? `${comparison.change_percent > 0 ? '+' : ''}${comparison.change_percent}%`
                  : '—'}
              </div>
              <div className="rp-trend-sub">
                Önceki: <strong>{comparison.previous_period_events ?? '—'}</strong> olay
                &nbsp;→&nbsp;
                Şu an: <strong>{summary?.total_events ?? '—'}</strong>
              </div>
            </>
          )}
        </div>

        {/* Lokasyon kartı */}
        <div className="rp-info-card">
          <div className="rp-info-label">En Kritik Bölge</div>
          {summaryLoading ? (
            <div className="rp-info-placeholder">…</div>
          ) : topLocation ? (
            <>
              <div className="rp-loc-zone">{topLocation.zone}</div>
              <div className="rp-loc-cam">{topLocation.camera_id}</div>
              <div className="rp-loc-count">
                <span>{topLocation.event_count}</span> olay
              </div>
            </>
          ) : (
            <div className="rp-trend-nodata">Bölge verisi yok</div>
          )}
        </div>

      </div>

      {/* ── Dağılım grafiği ── */}
      <div className="rp-chart-panel">
        {chartLoading ? (
          <div className="rp-loading">Yükleniyor…</div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4e" vertical={false} />
              <XAxis
                dataKey="label"
                tick={{ fill: '#7a8aa0', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                angle={period !== 'daily' ? -35 : 0}
                textAnchor={period !== 'daily' ? 'end' : 'middle'}
                interval={period === 'monthly' ? 2 : 0}
              />
              <YAxis tick={{ fill: '#7a8aa0', fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a4e', borderRadius: 6, fontSize: 12 }}
                labelStyle={{ color: '#c0cde0', marginBottom: 4 }}
                cursor={{ fill: 'rgba(255,255,255,0.03)' }}
              />
              <Legend wrapperStyle={{ fontSize: 12, color: '#7a8aa0', paddingTop: 12 }} formatter={val => LABELS[val] || val} />
              {Object.keys(COLORS).map(key => (
                <Bar key={key} dataKey={key} fill={COLORS[key]} stackId="a" radius={key === 'fire' ? [4,4,0,0] : [0,0,0,0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ── Kamera / bölge dağılımı ── */}
      {!summaryLoading && locations.length > 0 && (
        <div className="rp-chart-panel">
          <div className="rp-section-title">Bölge Dağılımı</div>
          <div className="rp-loc-list">
            {locations.map(loc => {
              const pct = Math.round((loc.event_count / (summary?.total_events || 1)) * 100)
              return (
                <div key={loc.camera_id} className="rp-loc-row">
                  <div className="rp-loc-row-label">
                    <span className="rp-loc-row-zone">{loc.zone}</span>
                    <span className="rp-loc-row-cam">{loc.camera_id}</span>
                  </div>
                  <div className="rp-loc-row-bar-wrap">
                    <div className="rp-loc-row-bar-fill" style={{ width: `${pct}%` }} />
                  </div>
                  <div className="rp-loc-row-count">{loc.event_count}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ── LLM Raporu ── */}
      <div className="rp-llm-panel">
        <div className="rp-llm-header">
          <span className="rp-section-title">AI Güvenlik Raporu</span>
          <button
            className="rp-llm-btn"
            onClick={handleGenerateLLM}
            disabled={llmLoading || summaryLoading || !summary}
          >
            {llmLoading ? <span className="rp-spinner" /> : null}
            {llmLoading ? 'Oluşturuluyor…' : 'Raporu Oluştur'}
          </button>
        </div>
        {llmText && (
          <div
            className="rp-llm-text"
            dangerouslySetInnerHTML={{ __html: llmText.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br/>') }}
          />
        )}
        {llmError && <div className="rp-llm-error">{llmError}</div>}
        {!llmText && !llmError && !llmLoading && (
          <div className="rp-llm-empty">
            Rapor oluşturmak için butona tıklayın. Model yanıtı ~15-30 saniye sürebilir.
          </div>
        )}
      </div>

    </div>
  )
}
