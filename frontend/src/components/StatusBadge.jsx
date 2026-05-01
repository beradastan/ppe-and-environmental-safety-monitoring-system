import './StatusBadge.css'

export default function StatusBadge({ status, viewed = false }) {
  if (status !== 'new' || viewed) return null
  return <span className="badge badge--new">YENİ</span>
}
