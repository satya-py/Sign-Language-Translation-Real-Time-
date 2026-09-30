import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { useTranslator } from '../useTranslator.js'
import { speak } from '../speech.js'
import { useApp } from '../store.jsx'
import { apiUrl, wsUrl } from '../api.js'

/**
 * Three-person call: deaf user, interpreter, hearing person.
 *
 * Media stays peer to peer — the backend only relays messages — so this works
 * between two browsers on the same machine or LAN. With a single browser open you
 * still see your own camera plus the AI captions, which is what a demo needs.
 *
 * The AI keeps translating during the call and its captions go to everyone, so the
 * interpreter can correct the machine rather than replace it.
 */
export default function InterpreterCall() {
  const { roomId } = useParams()
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const { profile } = useApp()
  const selfVideo = useRef(null)
  const t = useTranslator(selfVideo)
  const roomSocket = useRef(null)

  const [peers, setPeers] = useState(1)
  const [captions, setCaptions] = useState([])
  const [chat, setChat] = useState([])
  const [message, setMessage] = useState('')
  const [muted, setMuted] = useState(false)
  const [cameraOn, setCameraOn] = useState(false)
  const [captionsOn, setCaptionsOn] = useState(true)
  const [aiOn, setAiOn] = useState(true)
  const [role, setRole] = useState(params.get('role') || 'user')
  const kind = params.get('kind') || 'healthcare'

  // Room channel: chat, captions and presence.
  useEffect(() => {
    const socket = new WebSocket(wsUrl(`/api/interpreter/ws/${roomId}`))
    roomSocket.current = socket
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'joined' || data.type === 'peer-left') setPeers(data.peers)
      if (data.type === 'chat') setChat((prev) => [...prev, data])
      if (data.type === 'caption') setCaptions((prev) => [...prev, data])
    }
    return () => socket.close()
  }, [roomId])

  // Share every finished AI sentence with the room.
  useEffect(() => {
    if (!aiOn || !t.sentence) return
    const entry = { type: 'caption', who: profile?.name || 'Deaf user', text: t.sentence,
                    confidence: t.last?.prob, at: new Date().toLocaleTimeString() }
    setCaptions((prev) => (prev.at(-1)?.text === entry.text ? prev : [...prev, entry]))
    roomSocket.current?.readyState === 1 && roomSocket.current.send(JSON.stringify(entry))
    fetch(apiUrl(`/api/interpreter/rooms/${roomId}/caption`), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ who: entry.who, text: entry.text, confidence: entry.confidence }),
    }).catch(() => {})
  }, [t.sentence, t.last, aiOn, roomId, profile])

  const startCamera = async () => {
    try {
      await t.startCamera()
      setCameraOn(true)
    } catch (err) {
      setChat((prev) => [...prev, { who: 'system', text: `Camera blocked: ${err.message}` }])
    }
  }

  const sendChat = () => {
    const text = message.trim()
    if (!text) return
    const entry = { type: 'chat', who: profile?.name || 'You', text,
                    at: new Date().toLocaleTimeString() }
    setChat((prev) => [...prev, entry])
    roomSocket.current?.readyState === 1 && roomSocket.current.send(JSON.stringify(entry))
    setMessage('')
  }

  const leave = () => {
    t.stopCamera()
    roomSocket.current?.close()
    navigate('/app/interpreter')
  }

  const panelStyle = { position: 'relative', borderRadius: 'var(--r-lg)', overflow: 'hidden',
                       background: '#0b1020', aspectRatio: '4 / 3' }
  const tagStyle = { position: 'absolute', left: 10, bottom: 10, background: 'rgba(15,23,42,.85)',
                     border: '1px solid rgba(255,255,255,.15)', color: '#fff',
                     borderRadius: 999, padding: '6px 12px', fontSize: 12, fontWeight: 700,
                     fontFamily: 'var(--font-display)' }

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    gap: 12, flexWrap: 'wrap' }}>
        <div>
          <h2 className="title">
            {kind === 'emergency' ? 'Emergency interpreter call' : 'Interpreter call'}
          </h2>
          <p className="lead" style={{ marginBottom: 8 }}>
            Room {roomId} · {peers} connected · AI translation {aiOn ? 'on' : 'off'}
          </p>
        </div>
        <div className="pillrow">
          {['user', 'interpreter', 'doctor'].map((r) => (
            <button key={r} className={r === role ? 'on' : ''} onClick={() => setRole(r)}>
              I am the {r === 'user' ? 'deaf user' : r}
            </button>
          ))}
        </div>
      </div>

      <div className="split">
        <section>
          <div className="card">
            {/* main panel: whoever is signing */}
            <div style={panelStyle}>
              <video ref={selfVideo} autoPlay playsInline muted={muted}
                     style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              {!cameraOn && (
                <div className="placeholder" style={{ position: 'absolute', inset: 0,
                                                      display: 'grid', placeItems: 'center',
                                                      color: '#c7d2fe' }}>
                  Start your camera to join the call.
                </div>
              )}
              <span style={tagStyle}>
                {role === 'user' ? 'DEAF USER · LIVE ISL' : role === 'interpreter'
                  ? 'INTERPRETER · ISL' : 'DOCTOR'}
              </span>
              <div className="hud">
                <span className={`pill ${t.state === 'signing' ? 'live' : ''}`}>{t.state}</span>
                {t.last && (
                  <span className="pill">
                    {t.last.label.replace(/_/g, ' ')} {Math.round(t.last.prob * 100)}%
                  </span>
                )}
              </div>
            </div>

            {/* the other two participants */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12,
                          marginTop: 12 }}>
              {['Interpreter', 'Doctor'].map((who) => (
                <div key={who} style={{ ...panelStyle, aspectRatio: '16 / 9' }}>
                  <div style={{ position: 'absolute', inset: 0, display: 'grid',
                                placeItems: 'center', color: '#9aa4c7', fontSize: 13,
                                textAlign: 'center', padding: 10 }}>
                    {peers > 1 ? `${who} connected` : `Waiting for ${who.toLowerCase()} to join`}
                  </div>
                  <span style={tagStyle}>{who.toUpperCase()}</span>
                </div>
              ))}
            </div>

            <div className="controls">
              {cameraOn
                ? <button className="btn-secondary" onClick={() => { t.stopCamera(); setCameraOn(false) }}>
                    Camera off
                  </button>
                : <button className="btn-primary" onClick={startCamera}>Camera on</button>}
              <button className="btn-secondary" onClick={() => setMuted((m) => !m)}>
                {muted ? 'Unmute' : 'Mute'}
              </button>
              <button className="btn-secondary" onClick={() => setCaptionsOn((c) => !c)}>
                {captionsOn ? 'Hide captions' : 'Show captions'}
              </button>
              <button className="btn-secondary" onClick={() => setAiOn((a) => !a)}>
                AI translation: {aiOn ? 'on' : 'off'}
              </button>
              <button className="btn-secondary"
                      onClick={() => navigator.clipboard?.writeText(location.href)}>
                Invite participant
              </button>
              <button className="btn-danger" onClick={leave}>Leave call</button>
            </div>
            <p className="note" style={{ marginBottom: 0 }}>
              “Invite participant” copies this room link. Open it in another browser or on
              another device on the same network to bring the interpreter in.
            </p>
          </div>
        </section>

        <section>
          <div className="card" aria-live="polite">
            <div className="label">AI live translation</div>
            <div className="sentence">{t.sentence}</div>
            {t.last && (
              <>
                <div className="label" style={{ marginTop: 10 }}>
                  Confidence — {Math.round(t.last.prob * 100)}%
                </div>
                <div className="bar"><span style={{ width: `${t.last.prob * 100}%` }} /></div>
              </>
            )}
            <div className="glosses" style={{ marginTop: 12 }}>
              {t.glosses.map((g, i) => (
                <span className="gloss" key={`${g}-${i}`}>{g.toUpperCase().replace(/_/g, ' ')}</span>
              ))}
            </div>
            <div className="controls">
              <button className="btn-secondary" onClick={() => speak(t.sentence)}>Speak</button>
              <button className="btn-secondary" onClick={t.finish}>Finish sentence</button>
            </div>
          </div>

          {captionsOn && (
            <div className="card">
              <div className="label">Transcript</div>
              <div className="chat">
                {captions.length === 0 && <p className="note">AI captions will appear here.</p>}
                {captions.map((c, i) => (
                  <div className="msg patient" key={i}>
                    <span className="who" aria-hidden="true">AI</span>
                    <div className="bubble">
                      {c.text}
                      <span className="time">
                        {c.at}{c.confidence ? ` · ${Math.round(c.confidence * 100)}%` : ''}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card">
            <div className="label">Chat</div>
            <div className="chat" style={{ maxHeight: 200 }}>
              {chat.map((c, i) => (
                <div className={`msg ${c.who === 'system' ? 'patient' : 'doctor'}`} key={i}>
                  <span className="who" aria-hidden="true">{(c.who || '?')[0]}</span>
                  <div className="bubble">{c.text}<span className="time">{c.at}</span></div>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
              <input type="text" value={message} placeholder="Message everyone…"
                     aria-label="Chat message" style={{ flex: 1 }}
                     onChange={(e) => setMessage(e.target.value)}
                     onKeyDown={(e) => e.key === 'Enter' && sendChat()} />
              <button className="btn-primary" onClick={sendChat}>Send</button>
            </div>
          </div>
        </section>
      </div>

      {kind === 'emergency' && (
        <div className="emergency" style={{ marginTop: 'var(--gutter)' }}>
          <span style={{ fontSize: 26 }} aria-hidden="true">⚠</span>
          <div>
            An interpreter provides communication assistance and is <b>not a substitute for
            emergency medical or professional services</b>.
          </div>
        </div>
      )}
    </>
  )
}
