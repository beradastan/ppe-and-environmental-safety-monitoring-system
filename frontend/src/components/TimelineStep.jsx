import SignatureSummary from './SignatureSummary.jsx'
import { imageUrl } from '../api.js'
import './TimelineStep.css'

function formatDateTime(iso) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('tr-TR', {
      day: 'numeric', month: 'long', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  } catch { return iso }
}

export default function TimelineStep({ step }) {
  const {
    event_id, event_status, timestamp,
    change_reason, signature, image_filename,
  } = step

  const sig = signature || {}
  const personIds = [
    ...(sig.helmet_missing_ids || []),
    ...(sig.vest_missing_ids   || []),
    ...(sig.mask_missing_ids   || []),
  ]
  const uniqueIds = [...new Set(personIds)]

  return (
    <div className={`tl-step tl-step--${event_status}`}>
      <div className="tl-step__dot" />
      <div className="tl-step__body">

        <div className="tl-step__header">
          <span className="tl-step__time">{formatDateTime(timestamp)}</span>
        </div>

        <SignatureSummary signature={signature} />

        {uniqueIds.length > 0 && (
          <div className="tl-step__persons">
            {uniqueIds.map(id => (
              <span key={id} className="tl-step__person-tag">
                Kişi #{id}
                {sig.helmet_missing_ids?.includes(id) && <em>Baret</em>}
                {sig.vest_missing_ids?.includes(id)   && <em>Yelek</em>}
                {sig.mask_missing_ids?.includes(id)   && <em>Maske</em>}
              </span>
            ))}
          </div>
        )}

        {image_filename && (
          <img
            className="tl-step__img"
            src={imageUrl(event_id, image_filename)}
            alt={`${event_id} ${event_status}`}
            loading="lazy"
          />
        )}

      </div>
    </div>
  )
}
