import { useEffect, useState } from 'react'
import { fetchConfig, updateConfig } from '../api.js'
import './Settings.css'

const FIRE_FILTER_FIELDS = [
  { key: 'fire_min_area_ratio', label: 'Min. Alan Oranı (kare %)', min: 0.001, max: 0.10, step: 0.001 },
  { key: 'fire_growth_factor',  label: 'Büyüme Faktörü',            min: 1.1,   max: 3.0,  step: 0.1   },
]

const TIME_FIELDS = [
  { key: 'new_confirm_sec', label: 'Alarm Onay Süresi (s)', min: 0.5, max: 10, step: 0.5 },
]

const TOGGLE_FIELDS = [
  { key: 'use_helmet', label: 'Baret Tespiti'  },
  { key: 'use_vest',   label: 'Yelek Tespiti'  },
  { key: 'use_mask',   label: 'Maske Tespiti'  },
  { key: 'use_fire',   label: 'Yangın + Duman Tespiti' },
]

export default function Settings() {
  const [cfg, setCfg]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving]   = useState(false)
  const [saved, setSaved]     = useState(false)

  useEffect(() => {
    fetchConfig()
      .then(setCfg)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  function handleChange(key, val) {
    setCfg(prev => ({ ...prev, [key]: val }))
    setSaved(false)
  }

  async function handleSave(e) {
    e.preventDefault()
    setSaving(true)
    try {
      await updateConfig(cfg)
      setSaved(true)
    } catch (err) {
      console.error(err)
    } finally {
      setSaving(false)
    }
  }

  if (loading) return <div className="settings-loading">Yükleniyor…</div>
  if (!cfg)    return <div className="settings-loading">Yapılandırma alınamadı.</div>

  return (
    <div className="settings-page">
      <form className="settings-form" onSubmit={handleSave}>

        <div className="settings-note">
          Değişiklikler kaydedilir; sistem yeniden başlatıldığında aktif olur.
        </div>

        <section className="settings-section">
          <h3 className="settings-section__title">Hangi PPE'ler Tespit Edilsin</h3>
          <div className="settings-toggles">
            {TOGGLE_FIELDS.map(f => (
              <label key={f.key} className="settings-toggle">
                <input
                  type="checkbox"
                  checked={!!cfg[f.key]}
                  onChange={e => handleChange(f.key, e.target.checked)}
                />
                <span>{f.label}</span>
              </label>
            ))}
          </div>
        </section>

        <section className="settings-section">
          <h3 className="settings-section__title">Zamanlama</h3>
          {TIME_FIELDS.map(f => (
            <SliderRow
              key={f.key}
              field={f}
              value={cfg[f.key]}
              onChange={v => handleChange(f.key, v)}
            />
          ))}
        </section>

        <section className="settings-section">
          <h3 className="settings-section__title">Yangın + Duman Filtresi</h3>
          <p className="settings-section__hint">
            Alan oranı: karenin en az bu yüzdesi kadar büyük alev → alarm.<br/>
            Büyüme faktörü: son yarı / ilk yarı &gt; bu değer ise büyüyen alev → alarm.
          </p>
          {FIRE_FILTER_FIELDS.map(f => (
            <SliderRow
              key={f.key}
              field={f}
              value={cfg[f.key]}
              onChange={v => handleChange(f.key, v)}
            />
          ))}
        </section>

        <div className="settings-footer">
          <button className="settings-save-btn" type="submit" disabled={saving}>
            {saving ? 'Kaydediliyor…' : 'Kaydet'}
          </button>
          {saved && <span className="settings-saved">✓ Kaydedildi</span>}
        </div>

      </form>
    </div>
  )
}

function SliderRow({ field, value, onChange }) {
  return (
    <div className="slider-row">
      <label className="slider-label">{field.label}</label>
      <input
        type="range"
        min={field.min}
        max={field.max}
        step={field.step}
        value={value ?? field.min}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="slider-input"
      />
      <span className="slider-val">{Number(value).toFixed(field.step < 0.01 ? 3 : field.step < 1 ? 2 : 0)}</span>
    </div>
  )
}
