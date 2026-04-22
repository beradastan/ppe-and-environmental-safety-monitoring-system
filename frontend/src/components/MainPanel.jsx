import { useState, useEffect } from 'react'
import Timeline from './Timeline.jsx'
import { addNote } from '../api.js'
import './MainPanel.css'

function formatDateTime(iso) {
  if (!iso) return '—'
  try { return new Date(iso).toLocaleString('tr-TR') }
  catch { return iso }
}

export default function MainPanel({ eventId, timeline, notes: initialNotes = [], loading }) {
  const [notes, setNotes]   = useState(initialNotes)
  const [text, setText]     = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => { setNotes(initialNotes) }, [initialNotes])

  async function handleSaveNote(e) {
    e.preventDefault()
    if (!text.trim() || !eventId) return
    setSaving(true)
    try {
      const res = await addNote(eventId, text.trim())
      setNotes(prev => [...prev, res.note])
      setText('')
    } catch (err) {
      console.error(err)
    } finally {
      setSaving(false)
    }
  }

  if (!eventId) {
    return (
      <main className="main-panel main-panel--empty">
        <p>← Sol panelden bir olay seçin</p>
      </main>
    )
  }

  return (
    <main className="main-panel">
      <div className="main-panel__header">
        <h2>{eventId}</h2>
        {loading && <span className="main-panel__loading">Yükleniyor…</span>}
      </div>
      <div className="main-panel__content">
        <Timeline steps={timeline} />

        <div className="main-panel__notes">
          <h3 className="notes__title">Operatör Notları</h3>
          {notes.length === 0 && (
            <p className="notes__empty">Henüz not eklenmemiş.</p>
          )}
          {notes.map((n, i) => (
            <div key={i} className="notes__item">
              <span className="notes__ts">{formatDateTime(n.timestamp)}</span>
              <p className="notes__text">{n.text}</p>
            </div>
          ))}
          <form className="notes__form" onSubmit={handleSaveNote}>
            <textarea
              className="notes__input"
              placeholder="Not ekle…"
              value={text}
              onChange={e => setText(e.target.value)}
              rows={2}
            />
            <button className="notes__btn" type="submit" disabled={saving || !text.trim()}>
              {saving ? 'Kaydediliyor…' : 'Kaydet'}
            </button>
          </form>
        </div>
      </div>
    </main>
  )
}
