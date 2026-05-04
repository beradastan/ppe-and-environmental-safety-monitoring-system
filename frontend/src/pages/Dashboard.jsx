import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import EventCard from '../components/EventCard.jsx'
import { fetchStats } from '../api.js'
import './Dashboard.css'

const DIST_COLORS = { Baret: '#ffd740', Yelek: '#ff8c40', Maske: '#66bbff', Yangın: '#ff5f5f' }

const CHART_STYLES = {
  dark: {
    tick:           '#6a7d96',
    tooltipBg:      '#1c2133',
    tooltipBorder:  '#2c3650',
    tooltipLabel:   '#d0d8e8',
    tooltipItem:    '#7ab8ff',
  },
  light: {
    tick:           '#64748b',
    tooltipBg:      '#ffffff',
    tooltipBorder:  '#e2e8f0',
    tooltipLabel:   '#1e293b',
    tooltipItem:    '#2563eb',
  },
}

export default function Dashboard({ onNavigate, onSelectEvent, theme = 'dark', socket }) {
  const cs = CHART_STYLES[theme] || CHART_STYLES.dark
  const [stats, setStats]     = useState(null)
  const [loading, setLoading] = useState(true)

  function loadStats() {
    fetchStats().then(setStats).catch(console.error).finally(() => setLoading(false))
  }

  useEffect(() => { loadStats() }, [])

  useEffect(() => {
    if (!socket) return
    socket.on('new_alert',    loadStats)
    socket.on('event_closed', loadStats)
    return () => {
      socket.off('new_alert',    loadStats)
      socket.off('event_closed', loadStats)
    }
  }, [socket])

  if (loading) return <div className="db-loading">Yükleniyor…</div>
  if (!stats)  return <div className="db-loading">Veri alınamadı.</div>

  const dist = stats.distribution || {}
  const chartData = [
    { name: 'Baret',  value: dist.helmet || 0 },
    { name: 'Yelek',  value: dist.vest   || 0 },
    { name: 'Maske',  value: dist.mask   || 0 },
    { name: 'Yangın', value: dist.fire   || 0 },
  ]

  return (
    <div className="dashboard-page">

      {/* Stat kartlar */}
      <div className="db-cards">
        <div className="db-card db-card--danger">
          <div className="db-card__value">{stats.active_alarms}</div>
          <div className="db-card__label">Aktif Alarm</div>
        </div>
        <div className="db-card db-card--warn">
          <div className="db-card__value">{stats.today_violations}</div>
          <div className="db-card__label">Bugünkü İhlal</div>
        </div>
        <div className="db-card">
          <div className="db-card__value">{stats.total_events}</div>
          <div className="db-card__label">Toplam Olay</div>
        </div>
        <div className="db-card">
          <div className="db-card__value">
            {chartData.reduce((a, b) => a.value > b.value ? a : b).name}
          </div>
          <div className="db-card__label">En Sık İhlal</div>
        </div>
      </div>

      <div className="db-bottom">
        {/* İhlal dağılım grafiği */}
        <div className="db-chart-panel">
          <h3 className="db-section-title">İhlal Dağılımı</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <XAxis dataKey="name" tick={{ fill: cs.tick, fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: cs.tick, fontSize: 12 }} axisLine={false} tickLine={false} allowDecimals={false} />
              <Tooltip
                contentStyle={{ background: cs.tooltipBg, border: `1px solid ${cs.tooltipBorder}`, borderRadius: 6 }}
                labelStyle={{ color: cs.tooltipLabel }}
                itemStyle={{ color: cs.tooltipItem }}
                cursor={{ fill: 'rgba(128,128,128,0.06)' }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {chartData.map(entry => (
                  <Cell key={entry.name} fill={DIST_COLORS[entry.name]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Son alarmlar */}
        <div className="db-recent-panel">
          <div className="db-recent-header">
            <h3 className="db-section-title">Son Alarmlar</h3>
            <button className="db-all-btn" onClick={() => onNavigate('alerts')}>
              Tümünü gör →
            </button>
          </div>
          <div className="db-recent-list">
            {(stats.recent || []).length === 0 && (
              <p className="db-empty">Henüz olay yok.</p>
            )}
            {(stats.recent || []).map(evt => (
              <EventCard
                key={evt.event_id}
                event={evt}
                selected={false}
                onClick={() => { onNavigate('alerts'); onSelectEvent(evt.event_id) }}
              />
            ))}
          </div>
        </div>
      </div>

    </div>
  )
}
