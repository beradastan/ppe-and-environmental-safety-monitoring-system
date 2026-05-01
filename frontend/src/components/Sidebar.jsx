import EventCard from './EventCard.jsx'
import './Sidebar.css'

export default function Sidebar({ events, selectedId, onSelect, loading, viewedIds = new Set() }) {
  return (
    <aside className="sidebar">
      <div className="sidebar__header">
        <h2>Olaylar</h2>
        <span className="sidebar__count">{events.length} kayıt</span>
      </div>

      <div className="sidebar__list">
        {loading && <div className="sidebar__msg">Yükleniyor…</div>}
        {!loading && events.length === 0 && (
          <div className="sidebar__msg">Henüz olay yok.</div>
        )}
        {events.map(evt => (
          <EventCard
            key={evt.event_id}
            event={evt}
            selected={selectedId === evt.event_id}
            onClick={() => onSelect(evt.event_id)}
            viewed={viewedIds.has(evt.event_id)}
          />
        ))}
      </div>
    </aside>
  )
}
