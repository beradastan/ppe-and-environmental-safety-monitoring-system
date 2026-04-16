import './StatusBadge.css'

const COLOR_MAP = {
  new:      'badge--new',
  update:   'badge--update',
  active:   'badge--active',
  resolved: 'badge--resolved',
  idle:     'badge--idle',
}

export default function StatusBadge({ status }) {
  const cls = COLOR_MAP[status] || 'badge--idle'
  return <span className={`badge ${cls}`}>{status?.toUpperCase()}</span>
}
