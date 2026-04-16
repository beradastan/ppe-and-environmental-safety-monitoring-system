import StatusBadge from './StatusBadge.jsx'
import SignatureSummary from './SignatureSummary.jsx'
import './EventCard.css'

function formatTime(iso) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleTimeString('tr-TR')
  } catch {
    return iso
  }
}

export default function EventCard({ event, selected, onClick }) {
  return (
    <div className={`event-card ${selected ? 'event-card--selected' : ''}`} onClick={onClick}>
      <div className="event-card__header">
        <span className="event-card__id">{event.event_id}</span>
        <StatusBadge status={event.event_status} />
      </div>
      <div className="event-card__meta">
        <span>{formatTime(event.timestamp)}</span>
        <span>{event.duration_sec?.toFixed(1)}s</span>
      </div>
      <SignatureSummary signature={event.signature} />
    </div>
  )
}
