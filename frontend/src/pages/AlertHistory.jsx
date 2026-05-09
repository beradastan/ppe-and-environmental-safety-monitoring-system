import { useState, useEffect, useCallback } from 'react'
import Sidebar from '../components/Sidebar.jsx'
import MainPanel from '../components/MainPanel.jsx'
import { fetchEvents, fetchEventTimeline } from '../api.js'
import './AlertHistory.css'

const VIOLATION_TYPES = [
  { value: '',       label: 'Tüm ihlaller' },
  { value: 'helmet', label: 'Baret'        },
  { value: 'vest',   label: 'Yelek'        },
  { value: 'mask',   label: 'Maske'        },
  { value: 'fire',   label: 'Yangın'       },
]

export default function AlertHistory({ initialSelectedId, onEventSelect, socket }) {
  const [events, setEvents]           = useState([])
  const [selectedId, setSelectedId]   = useState(initialSelectedId || null)
  const [timeline, setTimeline]       = useState([])
  const [notes, setNotes]             = useState([])
  const [listLoading, setListLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)

  const [filters, setFilters]   = useState({ date: '', violation_type: '', status: 'closed' })
  const [viewedIds, setViewedIds] = useState(() => new Set())

  const loadEvents = useCallback((f = filters) => {
    setListLoading(true)
    fetchEvents(f)
      .then(data => setEvents(data.events || []))
      .catch(console.error)
      .finally(() => setListLoading(false))
  }, [filters])

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

  // Event kapandığında listeyi yenile
  useEffect(() => {
    if (!socket) return
    function onClosed() { loadEvents(filters) }
    socket.on('event_closed', onClosed)
    return () => socket.off('event_closed', onClosed)
  }, [socket, filters])

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
    setViewedIds(prev => { const s = new Set(prev); s.add(id); return s })
  }

  function handleFilterChange(key, val) {
    setFilters(prev => ({ ...prev, [key]: val }))
  }

  function clearFilters() {
    setFilters({ date: '', violation_type: '', status: 'closed' })
  }

  const hasFilters = filters.date || filters.violation_type

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
          {hasFilters && (
            <button className="ah-clear-btn" onClick={clearFilters}>✕ Temizle</button>
          )}
        </div>

        <Sidebar
          events={events}
          selectedId={selectedId}
          onSelect={handleSelect}
          loading={listLoading}
          viewedIds={viewedIds}
        />
      </div>

      <MainPanel
        eventId={selectedId}
        eventStatus={events.find(e => e.event_id === selectedId)?.event_status}
        falsePositive={events.find(e => e.event_id === selectedId)?.false_positive}
        timeline={timeline}
        notes={notes}
        loading={detailLoading}
        onClose={() => loadEvents(filters)}
      />
    </div>
  )
}
