import StatusBadge from './StatusBadge.jsx'
import './ToastContainer.css'

function Toast({ toast, onDismiss }) {
  const { id, event_id, event_status, signature = {} } = toast
  const { helmet_missing_ids = [], vest_missing_ids = [], fire_detected = false } = signature

  const parts = []
  if (helmet_missing_ids.length) parts.push(`Baretsiz: ${helmet_missing_ids.map(i => `#${i}`).join(', ')}`)
  if (vest_missing_ids.length)   parts.push(`Yeleksiz: ${vest_missing_ids.map(i => `#${i}`).join(', ')}`)
  if (fire_detected)             parts.push('YANGIN!')

  return (
    <div className={`toast toast--${event_status}`}>
      <div className="toast__header">
        <strong>{event_id}</strong>
        <StatusBadge status={event_status} />
        <button className="toast__close" onClick={() => onDismiss(id)}>×</button>
      </div>
      {parts.length > 0 && (
        <div className="toast__body">{parts.join(' | ')}</div>
      )}
    </div>
  )
}

export default function ToastContainer({ toasts, onDismiss }) {
  if (toasts.length === 0) return null

  return (
    <div className="toast-container">
      {toasts.slice(-5).map(t => (
        <Toast key={t.id} toast={t} onDismiss={onDismiss} />
      ))}
    </div>
  )
}
