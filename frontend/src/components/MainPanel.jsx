import { useState, useEffect } from 'react'
import Timeline from './Timeline.jsx'
import { addNote, markFalsePositive } from '../api.js'
import './MainPanel.css'

function formatDateTime(iso) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('tr-TR', {
      day: 'numeric', month: 'long', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  } catch { return iso }
}

export default function MainPanel({ eventId, eventStatus, falsePositive, timeline, notes: initialNotes = [], loading, onClose }) {
  const [notes, setNotes]         = useState(initialNotes)
  const [text, setText]           = useState('')
  const [saving, setSaving]       = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [falseNote, setFalseNote] = useState('')
  const [closing, setClosing]     = useState(false)

  useEffect(() => { setNotes(initialNotes) }, [initialNotes])
  useEffect(() => { setConfirming(false); setFalseNote('') }, [eventId])

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

  async function handleFalseAlarm() {
    if (!eventId) return
    setClosing(true)
    try {
      if (falseNote.trim()) {
        const res = await addNote(eventId, falseNote.trim())
        setNotes(prev => [...prev, res.note])
      }
      await markFalsePositive(eventId)
      setConfirming(false)
      onClose?.()
    } catch (err) {
      console.error(err)
    } finally {
      setClosing(false)
    }
  }

  const canMarkFalse = !falsePositive && (
    eventStatus === 'new' || eventStatus === 'active' || eventStatus === 'closed'
  )

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
        {falsePositive && (
          <span className="false-positive-badge">Yanlış Tespit</span>
        )}
        {canMarkFalse && !confirming && (
          <button className="false-alarm-btn" onClick={() => setConfirming(true)}>
            {eventStatus === 'closed' ? 'Yanlış Tespit' : 'Yanlış Alarm'}
          </button>
        )}
      </div>
      {confirming && (
        <div className="false-alarm-confirm">
          <p className="false-alarm-confirm__label">Bu event yanlış alarm olarak kapatılacak.</p>
          <textarea
            className="notes__input false-alarm-confirm__note"
            placeholder="Açıklama ekle (opsiyonel)…"
            value={falseNote}
            onChange={e => setFalseNote(e.target.value)}
            rows={2}
          />
          <div className="false-alarm-confirm__actions">
            <button className="false-alarm-confirm__cancel" onClick={() => setConfirming(false)} disabled={closing}>
              İptal
            </button>
            <button className="false-alarm-confirm__ok" onClick={handleFalseAlarm} disabled={closing}>
              {closing ? 'Kapatılıyor…' : 'Onayla'}
            </button>
          </div>
        </div>
      )}
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
