export async function fetchEvents() {
  const res = await fetch('/api/events')
  if (!res.ok) throw new Error('Event listesi alınamadı')
  return res.json()
}

export async function fetchEventTimeline(eventId) {
  const res = await fetch(`/api/events/${eventId}`)
  if (!res.ok) throw new Error(`Timeline alınamadı: ${eventId}`)
  return res.json()
}

export function imageUrl(eventId, filename) {
  return `/api/images/${eventId}/${filename}`
}
