import TimelineStep from './TimelineStep.jsx'
import './Timeline.css'

export default function Timeline({ steps }) {
  if (!steps || steps.length === 0) {
    return <div className="tl-empty">Timeline verisi bulunamadı.</div>
  }

  return (
    <div className="timeline">
      {steps.map((step, i) => (
        <TimelineStep key={`${step.event_status}-${i}`} step={step} />
      ))}
    </div>
  )
}
