import { useState, useEffect, useRef } from 'react'

import { fetchPipelineStatus, startPipeline, stopPipeline } from '../api.js'
import './PipelineControl.css'

export default function PipelineControl() {
  const [sourcePath, setSourcePath] = useState('')
  const [running, setRunning]       = useState(false)
  const [busy, setBusy]       = useState(false)
  const [browsing, setBrowsing] = useState(false)
  const ivRef = useRef(null)

  // Pipeline durum poll (her 3 saniyede)
  useEffect(() => {
    poll()
    ivRef.current = setInterval(poll, 3000)
    return () => clearInterval(ivRef.current)
  }, [])

  async function poll() {
    try {
      const s = await fetchPipelineStatus()
      setRunning(s.running)
      if (s.running && s.source) setSourcePath(s.source)
    } catch {}
  }

  async function handleBrowse() {
    setBrowsing(true)
    try {
      const res  = await fetch('/api/pipeline/browse')
      const data = await res.json()
      if (data.path) setSourcePath(data.path)
    } catch (e) {
      console.error(e)
    } finally {
      setBrowsing(false)
    }
  }

  async function handleStart() {
    if (!sourcePath.trim()) return
    setBusy(true)
    try {
      const res = await startPipeline(sourcePath.trim())
      if (res.ok) setRunning(true)
      else alert(res.error || 'Başlatılamadı.')
    } catch (e) {
      console.error(e)
    } finally {
      setBusy(false)
    }
  }

  async function handleStop() {
    setBusy(true)
    try {
      await stopPipeline()
      setRunning(false)
    } catch (e) {
      console.error(e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="pc-panel">
      <div className="pc-top">
        <div className="pc-controls">
          <div className="pc-header">
            <h3 className="pc-title">Pipeline Kontrolü</h3>
            <span className={`pc-badge${running ? ' pc-badge--on' : ''}`}>
              {running ? 'Çalışıyor' : 'Bekleniyor'}
            </span>
          </div>

          <div className="pc-source-row">
            <input
              className="pc-path-input"
              type="text"
              value={sourcePath}
              onChange={e => setSourcePath(e.target.value)}
              placeholder="Video yolu veya kamera numarası (0, 1…)"
              disabled={running}
            />
            <button
              className="pc-browse-btn"
              onClick={handleBrowse}
              disabled={running || browsing}
            >
              {browsing ? '…' : 'Gözat'}
            </button>
          </div>

          <div className="pc-action-row">
            {!running ? (
              <button
                className="pc-btn pc-btn--start"
                onClick={handleStart}
                disabled={busy || !sourcePath.trim()}
              >
                {busy ? 'Başlatılıyor…' : 'Başlat'}
              </button>
            ) : (
              <button
                className="pc-btn pc-btn--stop"
                onClick={handleStop}
                disabled={busy}
              >
                {busy ? 'Durduruluyor…' : 'Durdur'}
              </button>
            )}
          </div>
        </div>

        <div className="pc-stream-box">
          <div className="pc-stream-placeholder">
            {running ? 'Görüntü ayrı pencerede açıldı' : 'Pipeline durdurulmuş'}
          </div>
        </div>
      </div>
    </div>
  )
}
