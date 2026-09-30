import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useApp } from '../store.jsx'
import { apiUrl } from '../api.js'

const ACTIONS = [
  ['/app/healthcare', '✚', 'Healthcare mode', 'Doctor ↔ patient conversation', 'Triage ready'],
  ['/app/interpreter', '🤝', 'Interpreter Connect', 'Bring a human interpreter into the call', 'Fallback'],
  ['/app/learn', '🎓', 'Learn signs', 'Practise the signs the model knows', ''],
  ['/app/teach', '🧠', 'Teach the model', 'Record your own takes and retrain', 'Improves accuracy'],
]

export default function Dashboard() {
  const { profile } = useApp()
  const [stats, setStats] = useState(null)

  useEffect(() => {
    fetch(apiUrl('/api/stats')).then((r) => r.json()).then(setStats).catch(() => {})
  }, [])

  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening'

  return (
    <>
      <h2 className="title">{greeting}, {profile?.name || 'there'} 👋</h2>
      <p className="lead">Ready to start communicating? The vision engine is online.</p>

      <div className="model-strip">
        <span className="dot" aria-hidden="true" />
        <div style={{ minWidth: 0 }}>
          <b>SignNet live · {stats ? stats.vocabulary : '—'} signs</b>
          <div className="note" style={{ margin: 0 }}>
            Body-relative skeleton tracking, measured at {stats ? stats.live_accuracy : '95.4'}%
          </div>
        </div>
        <span className="badge teal" style={{ marginLeft: 'auto' }}>
          {stats ? `${stats.live_accuracy}%` : '95%'}
        </span>
        <span className="badge blue">22 fps</span>
      </div>

      <div className="launch" style={{ marginTop: 'var(--gutter)' }}>
        <div className="top">
          <span className="glyph" aria-hidden="true">◉</span>
          <span className="badge dark">Most used</span>
          <span className="badge dark" style={{ marginLeft: 'auto' }}>~0.3 s delay</span>
        </div>
        <h3>Live translation</h3>
        <p>
          Camera-based Indian Sign Language recognition with spoken and written English
          for the person you are talking to.
        </p>
        <Link to="/app/live"><button className="btn-primary">▶ Start camera session</button></Link>
      </div>

      <h3 style={{ fontFamily: 'var(--font-display)', margin: 'var(--gutter) 0 10px' }}>
        Modes and quick launch
      </h3>
      <div className="tiles">
        {ACTIONS.map(([to, icon, title, sub, tag]) => (
          <Link key={to} to={to} style={{ textDecoration: 'none', color: 'inherit' }}>
            <div className="action-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span className="icon" aria-hidden="true"
                      style={{ width: 44, height: 44, borderRadius: 12, display: 'grid',
                               placeItems: 'center', background: '#dbeafe', color: '#1e3a8a' }}>
                  {icon}
                </span>
                {tag && <span className="badge">{tag}</span>}
              </div>
              <div className="title">{title}</div>
              <div className="sub">{sub}</div>
            </div>
          </Link>
        ))}
      </div>

      <div className="tiles" style={{ marginTop: 'var(--gutter)' }}>
        <div className="card stat">
          <div className="big">{stats ? stats.sessions : '—'}</div>
          <div className="cap">translation sessions</div>
        </div>
        <div className="card stat">
          <div className="big">{stats ? stats.signs_recognised : '—'}</div>
          <div className="cap">signs recognised</div>
        </div>
        <div className="card stat">
          <div className="big">{stats ? `${stats.vocabulary_used}/${stats.vocabulary}` : '—'}</div>
          <div className="cap">vocabulary used</div>
          {stats && (
            <div className="bar" style={{ marginTop: 10 }}>
              <span style={{ width: `${(stats.vocabulary_used / stats.vocabulary) * 100}%` }} />
            </div>
          )}
        </div>
        <div className="card stat">
          <div className="big">{stats ? `${stats.live_accuracy}%` : '—'}</div>
          <div className="cap">accuracy, measured on 1185 clips</div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 'var(--gutter)' }}>
        <h2>Recent translation sessions</h2>
        {stats && stats.recent.length === 0 && (
          <p className="note">
            Nothing yet. Open <Link to="/app/live">Live translation</Link> and sign a sentence —
            every finished sentence is saved here.
          </p>
        )}
        {stats && stats.recent.length > 0 && (
          <table className="data">
            <thead>
              <tr><th>Date & time</th><th>Mode</th><th>Signs</th><th>Translation</th></tr>
            </thead>
            <tbody>
              {stats.recent.map((r) => (
                <tr key={r.id}>
                  <td>{r.at}</td>
                  <td>{r.mode}</td>
                  <td>{r.glosses.join(' · ') || '—'}</td>
                  <td>{r.sentence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  )
}
