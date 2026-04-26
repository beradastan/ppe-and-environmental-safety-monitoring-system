import { useState, useEffect, useCallback } from 'react'
import Sidebar from '../components/Sidebar.jsx'
import MainPanel from '../components/MainPanel.jsx'
import { fetchEvents, fetchEventTimeline } from '../api.js'
import './AlertHistory.css'

const VIOLATION_TYPES = [
  { value: '',       label: 'Tüm ihlaller' },
  { value: 'helmet', label: '⛑ Baret'      },
  { value: 'vest',   label: '🦺 Yelek'      },
  { value: 'mask',   label: '😷 Maske'      },
  { value: 'fire',   label: '🔥 Yangın'     },
]
const STATUSES = [
  { value: '',       label: 'Tüm durumlar' },
  { value: 'new',    label: 'Yeni'          },
  { value: 'active', label: 'Aktif'         },
]

export default function AlertHistory({ initialSelectedId, onEventSelect, socket }) {
  const [events, setEvents]           = useState([])
  const [selectedId, setSelectedId]   = useState(initialSelectedId || null)
  const [timeline, setTimeline]       = useState([])
  const [notes, setNotes]             = useState([])
  const [listLoading, setListLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)

  const [filters, setFilters] = useState({ date: '', violation_type: '', status: '' })

  const loadEvents = useCallback((f = filters) => {
    setListLoading(true)
    fetchEvents(f)
      .then(data => setEvents(data.events || []))
      .catch(console.error)
      .finally(() => setListLoading(false))
  }, [])

  const loadTimeline = useCallback((id) => {
    setDetailLoading(true)
    fetchEventTimeline(id)
      .then(data => {
        setTimeline(data.timeline || [])
        setNotes(data.notes || [])
      })
      .catch(console.error)
      .finally(() => setDetailLoading(false))
  }, [])

  useEffect(() => { loadEvents(filters) }, [filters])

  // Socket.IO: yeni alert geldiğinde listeyi yenile
  useEffect(() => {
    if (!socket) return
    function onAlert() { loadEvents(filters) }
    socket.on('new_alert', onAlert)
    return () => socket.off('new_alert', onAlert)
  }, [socket, filters])

  // LLM raporu hazır olduğunda seçili event'in timeline'ını yenile
  useEffect(() => {
    if (!socket) return
    function onLlmUpdated({ event_id }) {
      if (event_id === selectedId) loadTimeline(event_id)
    }
    socket.on('llm_updated', onLlmUpdated)
    return () => socket.off('llm_updated', onLlmUpdated)
  }, [socket, selectedId, loadTimeline])

  // Dışarıdan seçim gelirse (Dashboard → Alerts)
  useEffect(() => {
    if (initialSelectedId && initialSelectedId !== selectedId) {
      handleSelect(initialSelectedId)
    }
  }, [initialSelectedId])

  function handleSelect(id) {
    setSelectedId(id)
    setTimeline([])
    setNotes([])
    loadTimeline(id)
    onEventSelect?.(id)
  }

  function handleFilterChange(key, val) {
    setFilters(prev => ({ ...prev, [key]: val }))
  }

  function clearFilters() {
    setFilters({ date: '', violation_type: '', status: '' })
  }

  const hasFilters = filters.date || filters.violation_type || filters.status

  return (
    <div className="alert-history">
      <div className="ah-sidebar">
        <div className="ah-filters">
          <input
            type="date"
            className="ah-filter-input"
            value={filters.date}
            onChange={e => handleFilterChange('date', e.target.value)}
          />
          <select
            className="ah-filter-input"
            value={filters.violation_type}
            onChange={e => handleFilterChange('violation_type', e.target.value)}
          >
            {VIOLATION_TYPES.map(t => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </select>
          <select
            className="ah-filter-input"
            value={filters.status}
            onChange={e => handleFilterChange('status', e.target.value)}
          >
            {STATUSES.map(s => (
              <option key={s.value} value={s.value}>{s.label}</option>
            ))}
          </select>
          {hasFilters && (
            <button className="ah-clear-btn" onClick={clearFilters}>✕ Temizle</button>
          )}
        </div>

        <Sidebar
          events={events}
          selectedId={selectedId}
          onSelect={handleSelect}
          loading={listLoading}
        />
      </div>

      <MainPanel
        eventId={selectedId}
        timeline={timeline}
        notes={notes}
        loading={detailLoading}
      />
    </div>
  )
}
