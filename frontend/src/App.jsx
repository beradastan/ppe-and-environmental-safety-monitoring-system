import { useEffect, useState, useCallback } from 'react'
import Sidebar from './components/Sidebar.jsx'
import MainPanel from './components/MainPanel.jsx'
import ToastContainer from './components/ToastContainer.jsx'
import { fetchEvents, fetchEventTimeline } from './api.js'
import socket from './socket.js'

function upsertEvent(list, newData) {
  const idx = list.findIndex(e => e.event_id === newData.event_id)
  if (idx === -1) {
    // Yeni event → listenin başına ekle
    return [newData, ...list]
  }
  // Mevcut event → güncelle
  const updated = [...list]
  updated[idx] = { ...updated[idx], ...newData }
  return updated
}

export default function App() {
  const [events, setEvents]       = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [timeline, setTimeline]   = useState([])
  const [toasts, setToasts]       = useState([])
  const [listLoading, setListLoading]     = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)

  // İlk yükleme
  useEffect(() => {
    fetchEvents()
      .then(data => setEvents(data.events || []))
      .catch(console.error)
      .finally(() => setListLoading(false))
  }, [])

  // Socket.IO: gerçek zamanlı alert
  useEffect(() => {
    function onAlert(data) {
      setEvents(prev => upsertEvent(prev, data))
      addToast(data)

      // Seçili event güncelleniyorsa timeline'ı da yenile
      if (data.event_id === selectedId) {
        loadTimeline(data.event_id)
      }
    }

    socket.on('new_alert', onAlert)
    return () => socket.off('new_alert', onAlert)
  }, [selectedId])

  const loadTimeline = useCallback((eventId) => {
    setDetailLoading(true)
    fetchEventTimeline(eventId)
      .then(data => setTimeline(data.timeline || []))
      .catch(console.error)
      .finally(() => setDetailLoading(false))
  }, [])

  function handleSelect(eventId) {
    setSelectedId(eventId)
    setTimeline([])
    loadTimeline(eventId)
  }

  function addToast(data) {
    const id = Date.now() + Math.random()
    setToasts(prev => [...prev, { id, ...data }])
    setTimeout(() => dismissToast(id), 5000)
  }

  function dismissToast(id) {
    setToasts(prev => prev.filter(t => t.id !== id))
  }

  return (
    <>
      <div className="dashboard">
        <Sidebar
          events={events}
          selectedId={selectedId}
          onSelect={handleSelect}
          loading={listLoading}
        />
        <MainPanel
          eventId={selectedId}
          timeline={timeline}
          loading={detailLoading}
        />
      </div>
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </>
  )
}
