import { useEffect, useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useApp } from '../store.jsx'
import { apiUrl } from '../api.js'

// The five places worth a thumb on a phone (the sidebar keeps everything).
const BOTTOM = [
  ['/app/dashboard', 'Dashboard', '▤'],
  ['/app/live', 'Translate', '◉'],
  ['/app/healthcare', 'Healthcare', '✚'],
  ['/app/learn', 'Learn', '🎓'],
  ['/app/settings', 'Settings', '⚙'],
]

const LINKS = [
  ['/app/dashboard', 'Dashboard', '▤'],
  ['/app/live', 'Live translation', '◉'],
  ['/app/healthcare', 'Healthcare', '✚'],
  ['/app/conversation', 'General conversation', '💬'],
  ['/app/fingerspell', 'Fingerspell', '🔤'],
  ['/app/learn', 'Learn signs', '🎓'],
  ['/app/teach', 'Teach the model', '🧠'],
  ['/app/interpreter', 'Interpreter Connect', '🤝'],
  ['/app/interpreter/dashboard', 'Interpreter desk', '🎧'],
  ['/app/history', 'History', '🕘'],
  ['/app/settings', 'Settings', '⚙'],
]

/** Signed-in app: sidebar + topbar, as in the design. */
export default function AppLayout() {
  const { profile, signOut } = useApp()
  const navigate = useNavigate()
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetch(apiUrl('/api/health')).then((r) => r.json()).then(setHealth).catch(() => setHealth(false))
  }, [])
  const name = profile?.name || 'Guest'
  const initials = name.split(' ').map((p) => p[0]).slice(0, 2).join('').toUpperCase()

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">SB</div>
          <div>
            <h1>SignBridge AI</h1>
            <p>Hospital edition</p>
          </div>
        </div>
        <nav>
          {LINKS.map(([to, label, icon]) => (
            <NavLink key={to} to={to} className={({ isActive }) => (isActive ? 'active' : '')}>
              <span aria-hidden="true">{icon}</span> {label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <div className="shell-main">
        <div className="topbar">
          <NavLink to="/" style={{ textDecoration: 'none', color: 'var(--ink-soft)', fontSize: 14 }}>
            ← Back to home
          </NavLink>
          <div className="who">
            <span className={`status-pill ${health === false ? 'off' : health ? '' : 'busy'}`}>
              {health === false ? 'Offline' : health ? 'Ready' : 'Starting…'}
            </span>
            <span className="avatar" aria-hidden="true">{initials || 'G'}</span>
            <span>{name}</span>
            <button className="btn-secondary" style={{ minHeight: 40 }}
                    onClick={() => { signOut(); navigate('/') }}>
              {profile ? 'Sign out' : 'Set name'}
            </button>
          </div>
        </div>
        <div className="page"><Outlet /></div>
      </div>

      <nav className="bottom-nav" aria-label="Main">
        {BOTTOM.map(([to, label, icon]) => (
          <NavLink key={to} to={to} className={({ isActive }) => (isActive ? 'active' : '')}>
            <span className="ic" aria-hidden="true">{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>
    </div>
  )
}
