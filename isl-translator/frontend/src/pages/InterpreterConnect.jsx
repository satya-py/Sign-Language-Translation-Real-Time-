import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useApp } from '../store.jsx'
import { apiUrl } from '../api.js'

const LANGUAGES = ['', 'Indian Sign Language', 'English', 'Hindi', 'Bengali', 'Tamil', 'Urdu']

function Avatar({ name, size = 52 }) {
  const initials = name.split(' ').map((p) => p[0]).slice(0, 2).join('')
  return (
    <div className="avatar" style={{ width: size, height: size, fontSize: size / 2.6 }}
         aria-hidden="true">{initials}</div>
  )
}

/** Card for one interpreter, used by the directory and the emergency screen. */
function InterpreterCard({ person, onInvite, compact }) {
  return (
    <div className="card" style={{ display: 'grid', gap: 12 }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <Avatar name={person.name} />
        <div style={{ minWidth: 0 }}>
          <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 17 }}>
            {person.name}
          </div>
          <div className="note" style={{ margin: 0 }}>{person.title}</div>
        </div>
        <span className={`chip ${person.available ? 'live' : ''}`} style={{ marginLeft: 'auto' }}>
          {person.available ? 'Available now' : 'Busy'}
        </span>
      </div>

      {!compact && (
        <>
          <div>
            <div className="label">Languages</div>
            <div className="pillrow">
              {person.languages.map((l) => <button key={l} type="button">{l}</button>)}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }} className="note">
            <span>{person.experience_years}+ years experience</span>
            <span>★ {person.rating}</span>
            <span>{person.calls} calls</span>
            {person.demo && <span>demo profile</span>}
          </div>
        </>
      )}

      <button className="btn-primary" disabled={!person.available}
              onClick={() => onInvite(person)}>
        {person.available ? 'Invite to conversation' : 'Unavailable'}
      </button>
    </div>
  )
}

/** Invitation modal: who, what kind of conversation, who else is in the room. */
function InviteModal({ person, onClose, onSent }) {
  const { profile } = useApp()
  const [kind, setKind] = useState('healthcare')
  const [note, setNote] = useState('')
  const [sending, setSending] = useState(false)

  const send = async () => {
    setSending(true)
    const body = {
      interpreter_id: person.id,
      kind,
      from: profile?.name || 'Guest',
      participants: kind === 'healthcare' ? ['Deaf user', 'Doctor'] : ['Deaf user', 'Other person'],
      note,
    }
    const request = await (await fetch(apiUrl('/api/interpreter/request'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    })).json()
    setSending(false)
    onSent(request)
  }

  return (
    <div role="dialog" aria-modal="true" aria-label="Invite interpreter"
         style={{ position: 'fixed', inset: 0, background: 'rgba(15,23,42,.55)',
                  display: 'grid', placeItems: 'center', padding: 16, zIndex: 50 }}
         onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="card" style={{ maxWidth: 520, width: '100%', boxShadow: 'var(--level-2)' }}>
        <h2>Invite interpreter</h2>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16 }}>
          <Avatar name={person.name} />
          <div>
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700 }}>{person.name}</div>
            <div className="note" style={{ margin: 0 }}>
              {person.title} · {person.languages.join(' · ')}
            </div>
          </div>
        </div>

        <div className="label">Conversation type</div>
        <div className="pillrow" style={{ marginBottom: 14 }}>
          {['general', 'healthcare', 'emergency'].map((k) => (
            <button key={k} className={k === kind ? 'on' : ''} onClick={() => setKind(k)}>{k}</button>
          ))}
        </div>

        <div className="label">Participants</div>
        <p className="note" style={{ marginTop: 4 }}>
          {profile?.name || 'You'} (deaf user) · {kind === 'healthcare' ? 'Doctor' : 'Other person'}
        </p>

        <div className="field">
          <label htmlFor="note">Message to the interpreter</label>
          <input id="note" type="text" value={note} onChange={(e) => setNote(e.target.value)}
                 placeholder="Would you like to join this conversation as an interpreter?" />
        </div>

        <div className="controls">
          <button className="btn-primary" onClick={send} disabled={sending}>
            {sending ? 'Sending…' : 'Send invitation'}
          </button>
          <button className="btn-secondary" onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>
  )
}

/** Screen 1: the two entry points, normal and emergency. */
export function InterpreterConnect() {
  const navigate = useNavigate()
  const [stats, setStats] = useState(null)

  useEffect(() => {
    fetch(apiUrl('/api/interpreter/stats')).then((r) => r.json()).then(setStats).catch(() => {})
  }, [])

  return (
    <>
      <h2 className="title">Connect with an ISL interpreter</h2>
      <p className="lead">
        Get help from a human sign-language interpreter during your conversation.
        The AI keeps translating alongside them.
      </p>

      <div className="two-col">
        <div className="card">
          <div className="icon teal" aria-hidden="true"
               style={{ width: 44, height: 44, borderRadius: 12, display: 'grid',
                        placeItems: 'center', background: '#ccfbf1', color: '#0f766e' }}>
            💬
          </div>
          <h2 style={{ marginTop: 12 }}>Need help communicating?</h2>
          <p className="note">
            Connect with an available ISL interpreter for an ordinary conversation —
            a consultation, a form, a difficult explanation.
          </p>
          <button className="btn-primary" onClick={() => navigate('/app/interpreter/find')}>
            Find an interpreter
          </button>
          {stats && (
            <p className="note" style={{ marginBottom: 0 }}>
              {stats.available} of {stats.interpreters} interpreters available now.
            </p>
          )}
        </div>

        <div className="card" style={{ borderColor: '#fecdd3', background: '#fff5f6' }}>
          <div aria-hidden="true"
               style={{ width: 44, height: 44, borderRadius: 12, display: 'grid',
                        placeItems: 'center', background: '#ffe4e6', color: '#9f1239' }}>
            ☎
          </div>
          <h2 style={{ marginTop: 12, color: '#9f1239' }}>Emergency communication</h2>
          <p className="note">
            Quickly connect with any interpreter who is on call. Use this when something
            must be understood immediately.
          </p>
          <button className="btn-danger" style={{ background: '#e11d48', color: '#fff',
                                                  borderColor: '#e11d48' }}
                  onClick={() => navigate('/app/interpreter/emergency')}>
            Connect now
          </button>
        </div>
      </div>

      <div className="card" style={{ marginTop: 'var(--gutter)' }}>
        <div className="label">How this fits with the AI</div>
        <p className="note" style={{ marginBottom: 0 }}>
          The AI translator keeps running during an interpreter call, and its captions are
          shown to everyone. The interpreter corrects and fills the gaps — unusual vocabulary,
          consent, anything the model has not been taught. A human interpreter is the fallback,
          not a replacement, and the AI is never the authority in a medical decision.
        </p>
      </div>
    </>
  )
}

/** Screen 2: the directory with search and filters. */
export function FindInterpreter() {
  const navigate = useNavigate()
  const [people, setPeople] = useState([])
  const [query, setQuery] = useState('')
  const [language, setLanguage] = useState('')
  const [availability, setAvailability] = useState('all')
  const [inviting, setInviting] = useState(null)

  useEffect(() => {
    const params = new URLSearchParams({ q: query, language, availability })
    fetch(apiUrl(`/api/interpreter/list?${params}`)).then((r) => r.json()).then(setPeople).catch(() => {})
  }, [query, language, availability])

  return (
    <>
      <h2 className="title">Available interpreters</h2>
      <p className="lead">Invite someone into your conversation. They join as a third participant.</p>

      <div className="card">
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <input type="search" placeholder="Search by name or language…" value={query}
                 aria-label="Search interpreters" style={{ flex: 1, minWidth: 220 }}
                 onChange={(e) => setQuery(e.target.value)} />
          <select value={language} onChange={(e) => setLanguage(e.target.value)}
                  aria-label="Filter by language"
                  style={{ minHeight: 48, borderRadius: 12, padding: '0 12px',
                           border: '1.5px solid var(--line-strong)' }}>
            {LANGUAGES.map((l) => <option key={l} value={l}>{l || 'Any language'}</option>)}
          </select>
        </div>
        <div className="pillrow" style={{ marginTop: 12 }}>
          {['all', 'available', 'emergency'].map((a) => (
            <button key={a} className={a === availability ? 'on' : ''}
                    onClick={() => setAvailability(a)}>
              {a === 'all' ? 'Everyone' : a === 'available' ? 'Available now' : 'On call for emergencies'}
            </button>
          ))}
        </div>
      </div>

      <div className="tiles" style={{ marginTop: 'var(--gutter)' }}>
        {people.map((p) => (
          <InterpreterCard key={p.id} person={p} onInvite={setInviting} />
        ))}
        {people.length === 0 && <p className="note">No interpreter matches those filters.</p>}
      </div>

      {inviting && (
        <InviteModal person={inviting} onClose={() => setInviting(null)}
                     onSent={(request) => {
                       setInviting(null)
                       navigate(`/app/interpreter/call/${request.room}?req=${request.id}`)
                     }} />
      )}
    </>
  )
}

/** Screen 5: emergency — searches, then offers the first interpreter on call. */
export function EmergencyInterpreter() {
  const navigate = useNavigate()
  const { profile } = useApp()
  const [phase, setPhase] = useState('searching')
  const [person, setPerson] = useState(null)

  useEffect(() => {
    let cancelled = false
    const find = async () => {
      const list = await (await fetch(apiUrl('/api/interpreter/list?availability=emergency'))).json()
      if (cancelled) return
      setTimeout(() => {
        if (cancelled) return
        if (list.length) { setPerson(list[0]); setPhase('found') } else setPhase('none')
      }, 1600)                                   // brief search, so the state is visible
    }
    find()
    return () => { cancelled = true }
  }, [])

  const join = async () => {
    const request = await (await fetch(apiUrl('/api/interpreter/request'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ interpreter_id: person.id, kind: 'emergency',
                             from: profile?.name || 'Guest',
                             participants: ['Deaf user', 'Doctor'] }),
    })).json()
    await fetch(apiUrl(`/api/interpreter/requests/${request.id}/respond`), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ accept: true, interpreter_id: person.id }),
    })
    navigate(`/app/interpreter/call/${request.room}?kind=emergency`)
  }

  return (
    <>
      <div className="card" style={{ background: '#e11d48', color: '#fff', border: 0 }}>
        <h2 style={{ color: '#fff', margin: 0 }}>Emergency communication</h2>
        <p style={{ margin: '6px 0 0', opacity: .9 }}>
          Connecting you to an interpreter who is on call right now.
        </p>
      </div>

      <div className="card" style={{ marginTop: 'var(--gutter)', textAlign: 'center' }}>
        {phase === 'searching' && (
          <>
            <div className="sentence" style={{ minHeight: 0 }}>Finding an available ISL interpreter…</div>
            <div className="bar" style={{ marginTop: 16 }}>
              <span style={{ width: '60%', animation: 'sweep 1.2s ease-in-out infinite alternate' }} />
            </div>
          </>
        )}

        {phase === 'none' && (
          <>
            <div className="sentence" style={{ minHeight: 0 }}>No interpreter is on call.</div>
            <p className="note">
              Call your hospital&rsquo;s interpreter desk, and keep using AI translation meanwhile.
            </p>
            <button className="btn-secondary" onClick={() => navigate('/app/healthcare')}>
              Back to healthcare mode
            </button>
          </>
        )}

        {phase === 'found' && person && (
          <>
            <div className="chip live" style={{ marginBottom: 12 }}>Interpreter found</div>
            <div style={{ display: 'flex', gap: 14, alignItems: 'center', justifyContent: 'center' }}>
              <Avatar name={person.name} size={64} />
              <div style={{ textAlign: 'left' }}>
                <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 20 }}>
                  {person.name}
                </div>
                <div className="note" style={{ margin: 0 }}>Available now</div>
                <div className="note" style={{ margin: 0 }}>{person.languages.join(' • ')}</div>
              </div>
            </div>
            <div className="controls" style={{ justifyContent: 'center', marginTop: 18 }}>
              <button className="btn-primary" onClick={join}>Join emergency call</button>
              <button className="btn-secondary" onClick={() => navigate('/app/interpreter')}>
                Cancel
              </button>
            </div>
          </>
        )}
      </div>

      <div className="emergency" style={{ marginTop: 'var(--gutter)' }}>
        <span style={{ fontSize: 26 }} aria-hidden="true">⚠</span>
        <div>
          An interpreter provides communication assistance and is <b>not a substitute for
          emergency medical or professional services</b>. In a medical emergency, call your
          local emergency number first.
        </div>
      </div>
    </>
  )
}
