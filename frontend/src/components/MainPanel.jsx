import Timeline from './Timeline.jsx'
import './MainPanel.css'

export default function MainPanel({ eventId, timeline, loading }) {
  if (!eventId) {
    return (
      <main className="main-panel main-panel--empty">
        <p>← Sol panelden bir olay seçin</p>
      </main>
    )
  }

  return (
    <main className="main-panel">
      <div className="main-panel__header">
        <h2>{eventId}</h2>
        {loading && <span className="main-panel__loading">Yükleniyor…</span>}
      </div>
      <div className="main-panel__content">
        <Timeline steps={timeline} />
      </div>
    </main>
  )
}
