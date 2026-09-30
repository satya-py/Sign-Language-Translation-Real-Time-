import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useApp } from '../store.jsx'
import { apiUrl } from '../api.js'

/**
 * The other side of the feature: someone who takes interpreting shifts.
 *
 * Registering here puts a real person in the directory, so the "human interpreter"
 * in the demo can be an actual second person on another laptop rather than a
 * scripted animation.
 */
export default function InterpreterDashboard() {
  const { profile } = useApp()
  const navigate = useNavigate()
  const [me, setMe] = useState(() => {
    try { return JSON.parse(localStorage.getItem('signbridge.interpreter') || 'null') } catch { return null }
  })
  const [requests, setRequests] = useState([])
  const [stats, setStats] = useState(null)
  const [form, setForm] = useState({
    name: profile?.name || '', experience_years: 3,
    languages: ['Indian Sign Language', 'English', 'Hindi'],
  })

  const refresh = useCallback(() => {
    if (!me) return
    const params = new URLSearchParams({ interpreter_id: me.id, status: 'waiting' })
    fetch(apiUrl(`/api/interpreter/requests?${params}`)).then((r) => r.json()).then(setRequests).catch(() => {})
    fetch(apiUrl('/api/interpreter/stats')).then((r) => r.json()).then(setStats).catch(() => {})
  }, [me])

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, 4000)      // incoming requests arrive while you wait
    return () => clearInterval(timer)
  }, [refresh])

  const register = async () => {
    const record = await (await fetch(apiUrl('/api/interpreter/register'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...form, id: me?.id }),
    })).json()
    localStorage.setItem('signbridge.interpreter', JSON.stringify(record))
    setMe(record)
  }

  const setAvailable = async (available) => {
    const record = await (await fetch(apiUrl(`/api/interpreter/${me.id}/availability`), {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ available }),
    })).json()
    localStorage.setItem('signbridge.interpreter', JSON.stringify(record))
    setMe(record)
  }

  const respond = async (request, accept) => {
    await fetch(apiUrl(`/api/interpreter/requests/${request.id}/respond`), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ accept, interpreter_id: me.id }),
    })
    refresh()
    if (accept) navigate(`/app/interpreter/call/${request.room}?role=interpreter&kind=${request.kind}`)
  }

  if (!me) {
    return (
      <>
        <h2 className="title">Interpreter dashboard</h2>
        <p className="lead">Register to take interpreting requests from this network.</p>
        <div className="card" style={{ maxWidth: 560 }}>
          <div className="field">
            <label htmlFor="iname">Your name</label>
            <input id="iname" type="text" value={form.name}
                   onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </div>
          <div className="field">
            <label htmlFor="iexp">Years of experience</label>
            <input id="iexp" type="text" value={form.experience_years}
                   onChange={(e) => setForm({ ...form, experience_years: e.target.value })} />
          </div>
          <div className="field">
            <label htmlFor="ilang">Languages (comma separated)</label>
            <input id="ilang" type="text" value={form.languages.join(', ')}
                   onChange={(e) => setForm({ ...form,
                                              languages: e.target.value.split(',').map((s) => s.trim()) })} />
          </div>
          <button className="btn-primary" onClick={register} disabled={!form.name.trim()}>
            Register as an interpreter
          </button>
          <p className="note" style={{ marginBottom: 0 }}>
            Demo registration: your name and languages are stored on this machine so another
            device on the same network can invite you. No verification, no credentials.
          </p>
        </div>
      </>
    )
  }

  return (
    <>
      <h2 className="title">Interpreter dashboard</h2>
      <p className="lead">{me.name} · {me.languages.join(' · ')}</p>

      <div className="card">
        <div className="switch" style={{ borderBottom: 0 }}>
          <div>
            <b>Available for calls</b>
            <div className="note">
              {me.available ? 'You appear in the directory and receive requests.'
                : 'You are hidden from the directory.'}
            </div>
          </div>
          <button className="toggle" role="switch" aria-checked={me.available}
                  aria-label="Available for calls" onClick={() => setAvailable(!me.available)} />
        </div>
      </div>

      <div className="tiles" style={{ marginTop: 'var(--gutter)' }}>
        <div className="card stat">
          <div className="big">{stats ? stats.calls_completed : '—'}</div>
          <div className="cap">calls accepted</div>
        </div>
        <div className="card stat">
          <div className="big">{stats ? stats.active_calls : '—'}</div>
          <div className="cap">active sessions</div>
        </div>
        <div className="card stat">
          <div className="big">{stats ? `${stats.total_minutes}m` : '—'}</div>
          <div className="cap">total call time</div>
        </div>
        <div className="card stat">
          <div className="big">{stats ? stats.waiting_requests : '—'}</div>
          <div className="cap">waiting requests</div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 'var(--gutter)' }}>
        <h2>Incoming requests</h2>
        {requests.length === 0 && (
          <p className="note">
            Nothing waiting. Requests appear here within a few seconds of being sent.
          </p>
        )}
        {requests.map((r) => (
          <div key={r.id} className="card" style={{ boxShadow: 'none', marginBottom: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10,
                          flexWrap: 'wrap' }}>
              <div>
                <b>{r.from}</b>
                <div className="note" style={{ margin: 0 }}>
                  {r.kind === 'healthcare' ? 'Healthcare communication'
                    : r.kind === 'emergency' ? 'Emergency communication' : 'General conversation'}
                  {' · '}{r.participants.join(' + ')}
                </div>
                {r.note && <div className="note" style={{ margin: 0 }}>“{r.note}”</div>}
              </div>
              <span className={`chip ${r.kind === 'emergency' ? 'err' : ''}`}>{r.at}</span>
            </div>
            <div className="controls">
              <button className="btn-primary" onClick={() => respond(r, true)}>Accept</button>
              <button className="btn-secondary" onClick={() => respond(r, false)}>Decline</button>
            </div>
          </div>
        ))}
      </div>
    </>
  )
}
