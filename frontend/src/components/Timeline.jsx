import TimelineStep from './TimelineStep.jsx'
import './Timeline.css'

export default function Timeline({ steps }) {
  if (!steps || steps.length === 0) {
    return <div className="tl-empty">Zaman çizelgesi verisi bulunamadı.</div>
  }

  const step = steps.find(s => s.event_status === 'closed') || steps[steps.length - 1]

  return (
    <div className="timeline">
      <TimelineStep key={step.event_status} step={step} />
    </div>
  )
}
