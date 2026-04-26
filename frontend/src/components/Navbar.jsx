import './Navbar.css'

const TABS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'alerts',    label: 'Alarmlar'  },
  { id: 'reports',   label: 'Raporlar'  },
  { id: 'settings',  label: 'Ayarlar'   },
]

export default function Navbar({ page, onNavigate, activeAlarms = 0 }) {
  return (
    <nav className="navbar">
      <div className="navbar__brand">Factory Safety</div>
      <div className="navbar__tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`navbar__tab ${page === t.id ? 'navbar__tab--active' : ''}`}
            onClick={() => onNavigate(t.id)}
          >
            {t.label}
            {t.id === 'alerts' && activeAlarms > 0 && (
              <span className="navbar__badge">{activeAlarms}</span>
            )}
          </button>
        ))}
      </div>
    </nav>
  )
}
