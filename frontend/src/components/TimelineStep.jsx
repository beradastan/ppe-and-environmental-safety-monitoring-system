import StatusBadge from './StatusBadge.jsx'
import SignatureSummary from './SignatureSummary.jsx'
import { imageUrl } from '../api.js'
import './TimelineStep.css'

function formatDateTime(iso) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('tr-TR')
  } catch {
    return iso
  }
}

export default function TimelineStep({ step }) {
  const {
    event_id, event_status, timestamp, repeat_count,
    duration_sec, change_reason, signature, llm_report, image_filename,
  } = step

  return (
    <div className={`tl-step tl-step--${event_status}`}>
      <div className="tl-step__dot" />

      <div className="tl-step__body">
        <div className="tl-step__header">
          <StatusBadge status={event_status} />
          <span className="tl-step__time">{formatDateTime(timestamp)}</span>
          <span className="tl-step__meta">Tekrar: {repeat_count} | Süre: {duration_sec?.toFixed(1)}s</span>
        </div>

        {change_reason && change_reason !== 'initial_violation' && (
          <div className="tl-step__reason">Değişiklik: {change_reason}</div>
        )}

        <SignatureSummary signature={signature} />

        {image_filename && (
          <img
            className="tl-step__img"
            src={imageUrl(event_id, image_filename)}
            alt={`${event_id} ${event_status}`}
            loading="lazy"
          />
        )}

        {llm_report ? (
          <blockquote className="tl-step__llm">{llm_report}</blockquote>
        ) : (
          <p className="tl-step__no-llm">LLM raporu yok</p>
        )}
      </div>
    </div>
  )
}
