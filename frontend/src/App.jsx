import { useState, useEffect } from 'react'
import Navbar from './components/Navbar.jsx'
import ToastContainer from './components/ToastContainer.jsx'
import Dashboard from './pages/Dashboard.jsx'
import AlertHistory from './pages/AlertHistory.jsx'
import Reports from './pages/Reports.jsx'
import Settings from './pages/Settings.jsx'
import Demo from './pages/Demo.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import socket from './socket.js'

export default function App() {
  const [page, setPage]               = useState('dashboard')
  const [toasts, setToasts]           = useState([])
  const [activeAlarms, setActiveAlarms] = useState(0)
  const [pendingSelect, setPendingSelect] = useState(null)

  useEffect(() => {
    function onAlert(data) {
      addToast(data)
      if (data.event_status === 'new' || data.event_status === 'active') {
        setActiveAlarms(n => n + 1)
      }
    }
    function onResolved() {
      setActiveAlarms(n => Math.max(0, n - 1))
    }
    socket.on('new_alert', onAlert)
    socket.on('event_resolved', onResolved)
    return () => {
      socket.off('new_alert', onAlert)
      socket.off('event_resolved', onResolved)
    }
  }, [])

  function addToast(data) {
    const id = Date.now() + Math.random()
    setToasts(prev => [...prev, { id, ...data }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000)
  }

  function handleSelectFromDashboard(eventId) {
    setPendingSelect(eventId)
  }

  function navigateTo(p) {
    if (p !== 'alerts') setPendingSelect(null)
    setPage(p)
    if (p === 'alerts' && activeAlarms > 0) setActiveAlarms(0)
  }

  return (
    <div className="app-shell">
      <Navbar page={page} onNavigate={navigateTo} activeAlarms={activeAlarms} />
      <div className="app-content">
        <ErrorBoundary>
          {page === 'dashboard' && (
            <Dashboard
              onNavigate={navigateTo}
              onSelectEvent={handleSelectFromDashboard}
            />
          )}
          {page === 'alerts' && (
            <AlertHistory
              initialSelectedId={pendingSelect}
              socket={socket}
            />
          )}
          {page === 'reports'  && <Reports />}
          {page === 'settings' && <Settings />}
          {page === 'demo'     && <Demo />}
        </ErrorBoundary>
      </div>
      <ToastContainer toasts={toasts} onDismiss={id => setToasts(prev => prev.filter(t => t.id !== id))} />
    </div>
  )
}
