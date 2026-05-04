import { useState, useEffect } from 'react'
import Navbar from './components/Navbar.jsx'
import ToastContainer from './components/ToastContainer.jsx'
import Dashboard from './pages/Dashboard.jsx'
import AlertHistory from './pages/AlertHistory.jsx'
import Reports from './pages/Reports.jsx'
import Settings from './pages/Settings.jsx'
import CameraSetup from './pages/CameraSetup.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import socket from './socket.js'

const CAM_STATUS_MSG = {
  offline: 'Kamera bağlantısı kesildi!',
  frozen:  'Kamera görüntüsü dondu!',
  dark:    'Kamera görüntüsü karardı — lens kapalı veya ışık yok.',
}

export default function App() {
  const [page, setPage]               = useState('dashboard')
  const [toasts, setToasts]           = useState([])
  const [activeAlarms, setActiveAlarms] = useState(0)
  const [pendingSelect, setPendingSelect] = useState(null)
  const [camStatus, setCamStatus]     = useState('online')
  const [theme, setTheme]             = useState(() => {
    const saved = localStorage.getItem('theme') || 'dark'
    document.documentElement.setAttribute('data-theme', saved)
    return saved
  })

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
    socket.on('event_closed', onResolved)
    return () => {
      socket.off('new_alert', onAlert)
      socket.off('event_closed', onResolved)
    }
  }, [])

  useEffect(() => {
    function onCamStatus({ status }) { setCamStatus(status) }
    socket.on('camera_status', onCamStatus)
    return () => socket.off('camera_status', onCamStatus)
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

  function toggleTheme() {
    setTheme(t => {
      const next = t === 'dark' ? 'light' : 'dark'
      document.documentElement.setAttribute('data-theme', next)
      localStorage.setItem('theme', next)
      return next
    })
  }

  return (
    <div className="app-shell">
      <Navbar
        page={page}
        onNavigate={navigateTo}
        activeAlarms={activeAlarms}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
      {camStatus !== 'online' && (
        <div className={`cam-status-bar cam-status-bar--${camStatus}`}>
          <span className="cam-status-bar__dot" />
          {CAM_STATUS_MSG[camStatus] || camStatus}
          <button className="cam-status-bar__close" onClick={() => setCamStatus('online')}>✕</button>
        </div>
      )}
      <div className="app-content">
        <ErrorBoundary>
          {page === 'dashboard' && (
            <Dashboard
              onNavigate={navigateTo}
              onSelectEvent={handleSelectFromDashboard}
              theme={theme}
              socket={socket}
            />
          )}
          {page === 'alerts' && (
            <AlertHistory
              initialSelectedId={pendingSelect}
              socket={socket}
            />
          )}
          {page === 'reports'  && <Reports theme={theme} />}
          {page === 'settings' && <Settings />}
          {page === 'camera'   && <CameraSetup />}
        </ErrorBoundary>
      </div>
      <ToastContainer toasts={toasts} onDismiss={id => setToasts(prev => prev.filter(t => t.id !== id))} />
    </div>
  )
}
