/**
 * Where the backend is.
 *
 * Running locally the backend serves this page itself, so a plain "/api/..."
 * works and this file adds nothing. Hosted, the page is on Vercel and the
 * backend is somewhere else entirely - the same request then has to be sent to
 * that other origin, and the WebSocket has to go to wss:// there too.
 *
 * Set it at build time on Vercel:
 *
 *     VITE_API_BASE = https://your-backend.onrender.com
 *
 * No trailing slash. Leave it unset for local use.
 */

const configured = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '')

/** The API origin, or '' when the backend serves this page itself. */
export const API_BASE = configured

/** apiUrl('/api/health') -> 'https://backend.example.com/api/health' */
export function apiUrl(path) {
  if (!path.startsWith('/')) return `${API_BASE}/${path}`
  return `${API_BASE}${path}`
}

/** wsUrl('/ws') -> 'wss://backend.example.com/ws' */
export function wsUrl(path) {
  if (API_BASE) {
    return API_BASE.replace(/^http/, 'ws') + path
  }
  const proto = location.protocol === 'https:' ? 'wss://' : 'ws://'
  return proto + location.host + path
}

/** Same as fetch, but relative /api paths reach the configured backend. */
export function api(path, options) {
  return fetch(apiUrl(path), options)
}
