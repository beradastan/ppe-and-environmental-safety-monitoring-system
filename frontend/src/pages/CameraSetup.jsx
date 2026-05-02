import { useState, useEffect, useRef } from 'react'
import { fetchPipelineStatus, startPipeline, stopPipeline, browsePipeline } from '../api.js'
import './CameraSetup.css'

const CAM_ID_SUGGESTIONS = ['cam_01', 'cam_02', 'cam_03', 'cam_04']

export default function CameraSetup() {
  const [sourceTab, setSourceTab]     = useState('camera')   // 'camera' | 'video'
  const [camIndex, setCamIndex]       = useState('0')
  const [videoPath, setVideoPath]     = useState('')
  const [browsing, setBrowsing]       = useState(false)
  const [cameraId, setCameraId]       = useState('')
  const [zone, setZone]               = useState('')
  const [detMode, setDetMode]         = useState('crop')      // 'crop' | 'scene'
  const [devices, setDevices]         = useState([])   // MediaDeviceInfo[]
  const [status, setStatus]           = useState({ running: false, source: '', camera_id: '', zone: '', mode: '' })
  const [busy, setBusy]               = useState(false)
  const [error, setError]             = useState('')
  const pollRef = useRef(null)

  // Enumerate video devices via browser API
  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices()
      .then(devs => setDevices(devs.filter(d => d.kind === 'videoinput')))
      .catch(() => {})
  }, [])

  // Poll pipeline status every 3 s
  useEffect(() => {
    fetchPipelineStatus().then(setStatus).catch(() => {})
    pollRef.current = setInterval(
      () => fetchPipelineStatus().then(setStatus).catch(() => {}),
      3000,
    )
    return () => clearInterval(pollRef.current)
  }, [])

  const running = status.running

  function getSource() {
    return sourceTab === 'camera' ? camIndex : videoPath.trim()
  }

  async function handleBrowse() {
    setBrowsing(true)
    try {
      const res = await browsePipeline()
      if (res.path) setVideoPath(res.path)
    } catch {}
    finally { setBrowsing(false) }
  }

  async function handleStart() {
    setError('')
    const source = getSource()
    if (!source) { setError('Kaynak seçilmedi.'); return }
    if (sourceTab === 'video' && !source) { setError('Video yolu girilmedi.'); return }
    setBusy(true)
    try {
      const res = await startPipeline({
        source,
        camera_id: cameraId.trim(),
        zone: zone.trim(),
        mode: detMode,
      })
      if (!res.ok) setError(res.error || 'Başlatılamadı.')
      else setStatus(s => ({ ...s, running: true }))
    } catch (e) {
      setError('Sunucu hatası.')
    } finally {
      setBusy(false)
    }
  }

  async function handleStop() {
    setBusy(true)
    try {
      await stopPipeline()
      setStatus(s => ({ ...s, running: false, source: '', camera_id: '', zone: '' }))
    } catch {}
    finally { setBusy(false) }
  }

  return (
    <div className="cs-page">
      <div className="cs-header">
        <h2 className="cs-title">Kamera Kurulumu</h2>
        <span className={`cs-badge ${running ? 'cs-badge--on' : ''}`}>
          <span className="cs-dot" />
          {running ? 'Çalışıyor' : 'Bekleniyor'}
        </span>
      </div>

      {/* Running info banner */}
      {running && (
        <div className="cs-running-bar">
          <span>Aktif kaynak: <strong>{status.source}</strong></span>
          {status.camera_id && <span>Kamera ID: <strong>{status.camera_id}</strong></span>}
          {status.zone      && <span>Bölge: <strong>{status.zone}</strong></span>}
          {status.mode      && <span>Mod: <strong>{status.mode === 'crop' ? 'Crop-Based' : 'Scene-Based'}</strong></span>}
        </div>
      )}

      <div className="cs-body">
        {/* ── LEFT: source + settings ── */}
        <div className="cs-left">

          {/* Source card */}
          <div className="cs-card">
            <div className="cs-card__label">Kaynak</div>
            <div className="cs-tabs">
              <button
                className={`cs-tab ${sourceTab === 'camera' ? 'cs-tab--active' : ''}`}
                onClick={() => setSourceTab('camera')}
                disabled={running}
              >
                Kamera
              </button>
              <button
                className={`cs-tab ${sourceTab === 'video' ? 'cs-tab--active' : ''}`}
                onClick={() => setSourceTab('video')}
                disabled={running}
              >
                Video Dosyası
              </button>
            </div>

            {sourceTab === 'camera' ? (
              <div className="cs-cam-grid">
                {(devices.length > 0 ? devices : [{ deviceId: '0', label: '' }, { deviceId: '1', label: '' }])
                  .map((dev, idx) => {
                    const idxStr = String(idx)
                    const label  = dev.label || `Kamera ${idx}`
                    const selected = camIndex === idxStr
                    return (
                      <button
                        key={dev.deviceId || idx}
                        className={`cs-cam-card ${selected ? 'cs-cam-card--selected' : ''}`}
                        onClick={() => setCamIndex(idxStr)}
                        disabled={running}
                      >
                        <div className="cs-cam-icon">
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                            <path d="M23 7l-7 5 7 5V7z"/>
                            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
                          </svg>
                        </div>
                        <div className="cs-cam-name">{label}</div>
                        <div className="cs-cam-index">Indeks {idx}</div>
                      </button>
                    )
                  })}
              </div>
            ) : (
              <div className="cs-video-row">
                <input
                  className="cs-input"
                  type="text"
                  placeholder="Video dosyası yolu…"
                  value={videoPath}
                  onChange={e => setVideoPath(e.target.value)}
                  disabled={running}
                />
                <button
                  className="cs-browse-btn"
                  onClick={handleBrowse}
                  disabled={running || browsing}
                >
                  {browsing ? '…' : 'Gözat'}
                </button>
              </div>
            )}
          </div>

          {/* Detection mode card */}
          <div className="cs-card">
            <div className="cs-card__label">Detection Modu</div>
            <div className="cs-tabs">
              <button
                className={`cs-tab ${detMode === 'crop' ? 'cs-tab--active' : ''}`}
                onClick={() => setDetMode('crop')}
                disabled={running}
              >
                Crop-Based
              </button>
              <button
                className={`cs-tab ${detMode === 'scene' ? 'cs-tab--active' : ''}`}
                onClick={() => setDetMode('scene')}
                disabled={running}
              >
                Scene-Based
              </button>
            </div>
            <p className="cs-mode-desc">
              {detMode === 'crop'
                ? 'Kişi kırpılarak her PPE modeli ayrı çalışır. Daha hassas, kalabalık sahnelerde daha iyi.'
                : 'Tam kare üzerinde PPE tespiti yapılır. Daha hızlı, geniş açı sahnelerde etkili.'}
            </p>
          </div>

          {/* Settings card */}
          <div className="cs-card">
            <div className="cs-card__label">Kamera Tanımı</div>
            <div className="cs-field-row">
              <div className="cs-field">
                <label className="cs-field__label">Kamera ID</label>
                <div className="cs-field__row">
                  <input
                    className="cs-input"
                    type="text"
                    placeholder="cam_01"
                    value={cameraId}
                    onChange={e => setCameraId(e.target.value)}
                    disabled={running}
                    list="cs-camid-list"
                  />
                  <datalist id="cs-camid-list">
                    {CAM_ID_SUGGESTIONS.map(s => <option key={s} value={s} />)}
                  </datalist>
                </div>
              </div>

              <div className="cs-field">
                <label className="cs-field__label">Bölge</label>
                <input
                  className="cs-input"
                  type="text"
                  placeholder="ör. Üretim Hattı A"
                  value={zone}
                  onChange={e => setZone(e.target.value)}
                  disabled={running}
                />
              </div>
            </div>
          </div>

          {/* Error */}
          {error && <div className="cs-error">{error}</div>}

          {/* Action button */}
          {!running ? (
            <button
              className="cs-start-btn"
              onClick={handleStart}
              disabled={busy}
            >
              {busy ? 'Başlatılıyor…' : 'Sistemi Başlat'}
            </button>
          ) : (
            <button
              className="cs-stop-btn"
              onClick={handleStop}
              disabled={busy}
            >
              {busy ? 'Durduruluyor…' : 'Sistemi Durdur'}
            </button>
          )}
        </div>

        {/* ── RIGHT: info panel ── */}
        <div className="cs-right">
          <div className="cs-card cs-info-card">
            <div className="cs-card__label">Bilgi</div>
            <ul className="cs-info-list">
              <li>Kamera ID ve Bölge bilgileri isteğe bağlıdır; girilirse ihlal raporlarında konum analizi yapılabilir.</li>
              <li>Kamera indeksi: 0 genellikle dahili, 1+ harici USB kameralardır.</li>
              <li>Video modu: <code>.mp4</code>, <code>.avi</code>, <code>.mov</code>, <code>.mkv</code> formatları desteklenir.</li>
              <li>Pipeline başlatıldığında ayrı bir OpenCV penceresi açılır.</li>
            </ul>
          </div>

          <div className="cs-card cs-status-card">
            <div className="cs-card__label">Sistem Durumu</div>
            <div className="cs-status-rows">
              <div className="cs-status-row">
                <span className="cs-status-key">Pipeline</span>
                <span className={`cs-status-val ${running ? 'cs-status-val--on' : 'cs-status-val--off'}`}>
                  {running ? 'Çalışıyor' : 'Durdu'}
                </span>
              </div>
              {running && (
                <>
                  <div className="cs-status-row">
                    <span className="cs-status-key">Kaynak</span>
                    <span className="cs-status-val">{status.source}</span>
                  </div>
                  {status.camera_id && (
                    <div className="cs-status-row">
                      <span className="cs-status-key">Kamera ID</span>
                      <span className="cs-status-val">{status.camera_id}</span>
                    </div>
                  )}
                  {status.zone && (
                    <div className="cs-status-row">
                      <span className="cs-status-key">Bölge</span>
                      <span className="cs-status-val">{status.zone}</span>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
