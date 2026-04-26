const BASE = ''

async function _get(path) {
  const res = await fetch(BASE + path)
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`)
  return res.json()
}

async function _put(path, body) {
  const res = await fetch(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`PUT ${path} → ${res.status}`)
  return res.json()
}

async function _post(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`)
  return res.json()
}

export function fetchEvents(filters = {}) {
  const params = new URLSearchParams()
  if (filters.date)           params.set('date', filters.date)
  if (filters.violation_type) params.set('violation_type', filters.violation_type)
  if (filters.status)         params.set('status', filters.status)
  const qs = params.toString()
  return _get('/api/events' + (qs ? '?' + qs : ''))
}

export function fetchEventTimeline(eventId) {
  return _get(`/api/events/${eventId}`)
}

export function addNote(eventId, note) {
  return _post(`/api/events/${eventId}/note`, { note })
}

export function fetchStats() {
  return _get('/api/stats')
}

export function fetchReports(period, date) {
  const params = new URLSearchParams({ period })
  if (date) params.set('date', date)
  return _get('/api/reports?' + params.toString())
}

export function fetchConfig() {
  return _get('/api/config')
}

export function updateConfig(config) {
  return _put('/api/config', config)
}

export function imageUrl(eventId, filename) {
  return `/api/images/${eventId}/${filename}`
}
