import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { apiUrl } from './api.js'

/**
 * Demo profile + settings.
 *
 * The profile is deliberately local and password-free: this is a hackathon demo,
 * so we never collect or transmit a real credential. The name is only used to
 * greet the user and label sessions.
 *
 * Settings are stored on the backend, because the recognition threshold and pause
 * length actually change how the model is used.
 */
const Ctx = createContext(null)

const PROFILE_KEY = 'signbridge.profile'

function loadProfile() {
  try {
    return JSON.parse(localStorage.getItem(PROFILE_KEY) || 'null')
  } catch {
    return null
  }
}

export function AppProvider({ children }) {
  const [profile, setProfile] = useState(loadProfile)
  const [settings, setSettings] = useState(null)

  useEffect(() => {
    fetch(apiUrl('/api/settings')).then((r) => r.json()).then(setSettings).catch(() => {})
  }, [])

  useEffect(() => {
    if (!settings) return
    document.body.classList.toggle('large-text', !!settings.large_text)
    document.body.classList.toggle('high-contrast', !!settings.high_contrast)
    document.body.classList.toggle('reduced-motion', !!settings.reduced_motion)
  }, [settings])

  const value = useMemo(() => ({
    profile,
    settings,
    signIn(name, email) {
      const p = { name: name || 'Guest', email: email || '', since: new Date().toISOString() }
      localStorage.setItem(PROFILE_KEY, JSON.stringify(p))
      setProfile(p)
      return p
    },
    signOut() {
      localStorage.removeItem(PROFILE_KEY)
      setProfile(null)
    },
    async saveSettings(patch) {
      const next = { ...settings, ...patch }
      setSettings(next)                                  // instant UI feedback
      const r = await fetch(apiUrl('/api/settings'), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      })
      setSettings(await r.json())
    },
  }), [profile, settings])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export const useApp = () => useContext(Ctx)

/** Save a finished translation so History and the dashboard show real data. */
export function recordSession(entry) {
  return fetch(apiUrl('/api/sessions'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(entry),
  }).catch(() => {})
}
