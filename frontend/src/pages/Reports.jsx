import { useEffect, useState, useRef, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid,
} from 'recharts'
import { fetchReports, fetchReportSummary, generateReportLLM, fetchSavedReports, fetchSavedReport, downloadExportCSV, downloadExportPDF } from '../api.js'
import socket from '../socket.js'
import './Reports.css'

const PERIODS = [
  { value: 'daily',   label: 'Günlük'   },
  { value: 'weekly',  label: 'Haftalık'  },
  { value: 'monthly', label: 'Aylık'    },
]

const COLORS = { helmet: '#ffd740', vest: '#ff8c40', mask: '#66bbff', fire: '#ff5f5f' }
const LABELS = { helmet: 'Baret', vest: 'Yelek', mask: 'Maske', fire: 'Yangın' }

const CHART_STYLES = {
  dark: {
    tick:          '#6a7d96',
    tooltipBg:     '#1c2133',
    tooltipBorder: '#2c3650',
    tooltipLabel:  '#d0d8e8',
    grid:          '#2c3650',
    legend:        '#6a7d96',
  },
  light: {
    tick:          '#64748b',
    tooltipBg:     '#ffffff',
    tooltipBorder: '#e2e8f0',
    tooltipLabel:  '#1e293b',
    grid:          '#e2e8f0',
    legend:        '#64748b',
  },
}

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

function localDateStr(d) {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

function today() { return localDateStr(new Date()) }

function currentWeekStr() {
  const d = new Date()
  const day = (d.getDay() + 6) % 7
  const thu = new Date(d); thu.setDate(d.getDate() - day + 3)
  const yearStart = new Date(thu.getFullYear(), 0, 1)
  const wn = Math.ceil(((thu - yearStart) / 86400000 + 1) / 7)
  return `${thu.getFullYear()}-W${String(wn).padStart(2, '0')}`
}

function weekStrToMonday(ws) {
  const [yr, wp] = ws.split('-W')
  const year = parseInt(yr), week = parseInt(wp)
  const jan4 = new Date(year, 0, 4)
  const jan4Day = (jan4.getDay() + 6) % 7
  const mon = new Date(jan4)
  mon.setDate(4 - jan4Day + (week - 1) * 7)
  return localDateStr(mon)
}

function formatSavedDate(period, report_date) {
  if (period === 'weekly') {
    const s = new Date(report_date + 'T12:00:00')
    const e = new Date(s); e.setDate(s.getDate() + 6)
    const opts = { day: 'numeric', month: 'long' }
    return `${s.toLocaleDateString('tr-TR', opts)} – ${e.toLocaleDateString('tr-TR', opts)} ${s.getFullYear()}`
  }
  if (period === 'monthly') {
    return new Date(report_date + 'T12:00:00')
      .toLocaleDateString('tr-TR', { month: 'long', year: 'numeric' })
  }
  return new Date(report_date + 'T12:00:00')
    .toLocaleDateString('tr-TR', { day: 'numeric', month: 'long', year: 'numeric' })
}

export default function Reports({ theme = 'dark' }) {
  const cs = CHART_STYLES[theme] || CHART_STYLES.dark
  const [period, setPeriod]         = useState('weekly')
  const [dailyDate, setDailyDate]   = useState(today())
  const [weekStr,   setWeekStr]     = useState(currentWeekStr())
  const [monthStr,  setMonthStr]    = useState(today().slice(0, 7))
  const [chartData, setChartData]   = useState([])
  const [summary, setSummary]     = useState(null)
  const [llmText, setLlmText]     = useState('')
  const [chartLoading, setChartLoading]     = useState(true)
  const [summaryLoading, setSummaryLoading] = useState(true)
  const [llmLoading, setLlmLoading]         = useState(false)
  const [llmError, setLlmError]             = useState('')
  const pendingRef = useRef(null)

  const [savedReports, setSavedReports]     = useState([])
  const [savedLoading, setSavedLoading]     = useState(true)
  const [selectedSaved, setSelectedSaved]   = useState(null)
  const [exporting, setExporting]           = useState(false)

  const dateParam =
    period === 'daily'   ? dailyDate :
    period === 'weekly'  ? weekStrToMonday(weekStr) :
    monthStr + '-01'

  const loadSaved = useCallback((p = period) => {
    setSavedLoading(true)
    fetchSavedReports(p)
      .then(d => setSavedReports(d.reports || []))
      .catch(console.error)
      .finally(() => setSavedLoading(false))
  }, [period])

  useEffect(() => { setSavedReports([]); loadSaved(period); setSelectedSaved(null) }, [period])

  useEffect(() => {
    setChartLoading(true)
    fetchReports(period, dateParam)
      .then(res => setChartData(res.data || []))
      .catch(console.error)
      .finally(() => setChartLoading(false))
  }, [period, dailyDate, weekStr, monthStr])

  useEffect(() => {
    setSummaryLoading(true)
    setSummary(null)
    setLlmText('')
    setLlmError('')
    fetchReportSummary(period, dateParam)
      .then(setSummary)
      .catch(() => setSummary(null))
      .finally(() => setSummaryLoading(false))
  }, [period, dailyDate, weekStr, monthStr])

  useEffect(() => {
    if (!savedReports.length) return
    const match = savedReports.find(r => r.period === period && r.report_date === dateParam)
    if (match) {
      fetchSavedReport(match.id).then(d => setLlmText(d.llm_text || ''))
    } else {
      setLlmText('')
    }
  }, [dateParam, period, savedReports])

  useEffect(() => {
    function onReady({ period: p, date: d, llm_text, auto_generated }) {
      if (p === period) loadSaved(p)
      setSelectedSaved(prev => prev ? { ...prev, text: llm_text || '' } : prev)
      if (auto_generated) return
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
  }, [loadSaved])

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

  async function handleCSV() {
    setExporting(true)
    try {
      await downloadExportCSV(period, dateParam)
    } catch (err) {
      console.error('CSV export error:', err)
    } finally {
      setExporting(false)
    }
  }

  async function handlePDF() {
    setExporting(true)
    try {
      await downloadExportPDF(period, dateParam)
    } catch (err) {
      console.error('PDF export error:', err)
    } finally {
      setExporting(false)
    }
  }

  function handleSelectSaved(r) {
    if (selectedSaved?.id === r.id) return
    setSelectedSaved({ id: r.id, loading: true, text: '' })
    fetchSavedReport(r.id)
      .then(d => setSelectedSaved({ id: r.id, loading: false, text: d.llm_text || '' }))
      .catch(() => setSelectedSaved({ id: r.id, loading: false, text: 'Rapor yüklenemedi.' }))
  }

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

      {}
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
            value={dailyDate}
            max={today()}
            onChange={e => setDailyDate(e.target.value)}
          />
        )}
        {period === 'weekly' && (
          <input
            type="week"
            className="rp-date-input"
            value={weekStr}
            max={currentWeekStr()}
            onChange={e => setWeekStr(e.target.value)}
          />
        )}
        {period === 'monthly' && (
          <input
            type="month"
            className="rp-date-input"
            value={monthStr}
            max={today().slice(0, 7)}
            onChange={e => setMonthStr(e.target.value)}
          />
        )}
        <div className="rp-export-btns no-print">
          <button
            className="rp-export-btn"
            onClick={handleCSV}
            disabled={exporting}
            title="Olay verilerini CSV olarak indir"
          >
            {exporting ? '…' : 'CSV'}
          </button>
          <button
            className="rp-export-btn"
            onClick={handlePDF}
            disabled={exporting}
            title="Formatli PDF raporu indir"
          >
            {exporting ? '…' : 'PDF'}
          </button>
        </div>
      </div>

      {}
      <div className="rp-print-title">
        <strong>Fabrika Güvenlik Raporu</strong>
        <span>{PERIODS.find(p => p.value === period)?.label} — {dateParam || today()}</span>
      </div>

      {}
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

      {}
      <div className="rp-info-row">

        {}
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

        {}
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
                {comparison.change_percent != null
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

        {}
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

      {}
      <div className="rp-chart-panel">
        {chartLoading ? (
          <div className="rp-loading">Yükleniyor…</div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={cs.grid} vertical={false} />
              <XAxis
                dataKey="label"
                tick={{ fill: cs.tick, fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                angle={period !== 'daily' ? -35 : 0}
                textAnchor={period !== 'daily' ? 'end' : 'middle'}
                interval={period === 'monthly' ? 2 : 0}
              />
              <YAxis tick={{ fill: cs.tick, fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
              <Tooltip
                contentStyle={{ background: cs.tooltipBg, border: `1px solid ${cs.tooltipBorder}`, borderRadius: 6, fontSize: 12 }}
                labelStyle={{ color: cs.tooltipLabel, marginBottom: 4 }}
                cursor={{ fill: 'rgba(128,128,128,0.04)' }}
              />
              <Legend wrapperStyle={{ fontSize: 12, color: cs.legend, paddingTop: 12 }} formatter={val => LABELS[val] || val} />
              {Object.keys(COLORS).map(key => (
                <Bar key={key} dataKey={key} fill={COLORS[key]} stackId="a" radius={key === 'fire' ? [4,4,0,0] : [0,0,0,0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {}
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

      {}
      <div className="rp-llm-panel">
        <div className="rp-llm-header">
          <span className="rp-section-title">YZ Güvenlik Raporu</span>
          <button
            className="rp-llm-btn no-print"
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

      {}
      <div className="rp-saved-panel no-print">
        <div className="rp-section-title">
          Kaydedilmiş {PERIODS.find(p => p.value === period)?.label} Raporlar
        </div>
        {savedLoading ? (
          <div className="rp-loading">Yükleniyor…</div>
        ) : savedReports.length === 0 ? (
          <div className="rp-saved-empty">Henüz kaydedilmiş rapor yok.</div>
        ) : (
          <div className="rp-saved-layout">
            <ul className="rp-saved-list">
              {savedReports.map(r => (
                <li
                  key={r.id}
                  className={`rp-saved-row${selectedSaved?.id === r.id ? ' rp-saved-row--active' : ''}`}
                  onClick={() => handleSelectSaved(r)}
                >
                  <span className="rp-saved-row-label">
                    {formatSavedDate(r.period, r.report_date)}
                  </span>
                  <span className="rp-saved-row-right">
                    {r.auto_generated && <span className="rp-saved-auto">Otomatik</span>}
                    <span className="rp-saved-time">
                      {new Date(r.generated_at).toLocaleDateString('tr-TR', { day: 'numeric', month: 'long', year: 'numeric' })}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
            <div className="rp-saved-detail">
              {!selectedSaved && (
                <div className="rp-saved-empty">Soldan bir rapor seçin.</div>
              )}
              {selectedSaved?.loading && <span className="rp-spinner" />}
              {selectedSaved?.text && (
                <div
                  className="rp-llm-text"
                  dangerouslySetInnerHTML={{
                    __html: selectedSaved.text
                      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                      .replace(/\n/g, '<br/>')
                  }}
                />
              )}
            </div>
          </div>
        )}
      </div>

    </div>
  )
}

