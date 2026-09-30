import { useEffect, useRef, useState } from 'react'
import CameraPanel from '../components/CameraPanel.jsx'
import { speak } from '../speech.js'
import { recordSession } from '../store.jsx'
import { useTranslator } from '../useTranslator.js'
import { apiUrl } from '../api.js'

/**
 * Healthcare mode: patient signs on the left, doctor answers on the right.
 *
 * "Show in ISL" plays the real reference clips for the words in a phrase, so the
 * doctor's reply reaches a deaf patient who does not read English comfortably.
 * Only phrases whose signs exist in the library are offered.
 */
export default function Healthcare() {
  const videoRef = useRef(null)
  const t = useTranslator(videoRef)
  const [phrases, setPhrases] = useState([])
  const [reply, setReply] = useState('')
  const [shown, setShown] = useState(null)      // { text, signs }
  const [signIndex, setSignIndex] = useState(0)
  const [conversation, setConversation] = useState([])
  const lastSaved = useRef('')

  useEffect(() => {
    fetch(apiUrl('/api/phrases')).then((r) => r.json()).then(setPhrases).catch(() => {})
  }, [])

  // Only a finished sentence joins the conversation: the one on screen is rebuilt
  // after every sign so the patient can watch it grow.
  useEffect(() => {
    if (!t.finalSentence || !t.finalAt) return
    const key = `${t.finalAt}:${t.finalSentence}`
    if (key === lastSaved.current) return
    lastSaved.current = key
    setConversation((prev) => [...prev, { side: 'patient', text: t.finalSentence,
                                          at: new Date().toLocaleTimeString() }])
    recordSession({ mode: 'healthcare', sentence: t.finalSentence,
                   glosses: t.lastGlosses, source: t.source })
  }, [t.finalSentence, t.finalAt, t.lastGlosses, t.source])

  const say = (text, signs = []) => {
    if (!text) return
    setShown({ text, signs })
    setSignIndex(0)
    speak(text)
    setConversation((prev) => [...prev, { side: 'doctor', text, at: new Date().toLocaleTimeString() }])
  }

  const currentSign = shown?.signs?.[signIndex]

  return (
    <>
      <h2 className="title">Healthcare communication</h2>
      <p className="lead">Helping a deaf patient and a doctor understand each other.</p>

      <div className="two-col">
        {/* ------------------------------------------------ patient side */}
        <div className="card">
          <div className="label">Patient → Doctor</div>
          <CameraPanel videoRef={videoRef} translator={t} label="Patient camera" />

          <div className="label" style={{ marginTop: 18 }}>Detected signs</div>
          <div className="glosses">
            {t.glosses.map((g, i) => (
              <span className="gloss" key={`${g}-${i}`}>{g.toUpperCase().replace(/_/g, ' ')}</span>
            ))}
          </div>
          <div className="label" style={{ marginTop: 16 }}>Translated for the doctor</div>
          <div className="sentence">{t.sentence}</div>
        </div>

        {/* ------------------------------------------------- doctor side */}
        <div className="card">
          <div className="label">Doctor → Patient</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 8 }}>
            <input type="text" value={reply} placeholder="Type what you want to say…"
                   aria-label="Doctor's reply" style={{ flex: 1, minWidth: 200 }}
                   onChange={(e) => setReply(e.target.value)}
                   onKeyDown={(e) => { if (e.key === 'Enter') { say(reply.trim()); setReply('') } }} />
            <button className="btn-primary" onClick={() => { say(reply.trim()); setReply('') }}>
              Show
            </button>
          </div>

          <div className="reply-out">{shown?.text}</div>

          {shown?.signs?.length > 0 && (
            <>
              <div className="label">Shown in ISL — {currentSign?.replace(/_/g, ' ')}</div>
              <div className="stage" style={{ aspectRatio: '16 / 9' }}>
                <video key={currentSign} src={apiUrl(`/api/clip/${currentSign}`)} autoPlay muted loop
                       playsInline controls />
              </div>
              {shown.signs.length > 1 && (
                <div className="pillrow" style={{ marginTop: 10 }}>
                  {shown.signs.map((s, i) => (
                    <button key={s} className={i === signIndex ? 'on' : ''}
                            onClick={() => setSignIndex(i)}>
                      {s.replace(/_/g, ' ')}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
          {shown && shown.signs.length === 0 && (
            <p className="note">
              This sentence has no matching signs in the library yet, so the patient can read it
              on screen but not see it signed.
            </p>
          )}

          <div className="label" style={{ marginTop: 18 }}>Common medical phrases</div>
          {phrases.map((p) => (
            <button className="phrase-row" key={p.text} onClick={() => say(p.text, p.signs)}>
              <span>{p.text}</span>
              <span className="chip">{p.signs.length ? `${p.signs.length} signs` : 'text only'}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="card" style={{ marginTop: 'var(--gutter)' }}>
        <div className="label">Conversation</div>
        <div className="chat">
          {conversation.length === 0 && (
            <p className="note">The conversation will appear here as you both speak and sign.</p>
          )}
          {conversation.map((m, i) => (
            <div className={`msg ${m.side}`} key={i}>
              <span className="who" aria-hidden="true">{m.side === 'patient' ? 'ISL' : 'Dr'}</span>
              <div className="bubble">{m.text}<span className="time">{m.at}</span></div>
            </div>
          ))}
        </div>
      </div>

      <div className="emergency" style={{ marginTop: 'var(--gutter)' }}>
        <span style={{ fontSize: 28 }} aria-hidden="true">☎</span>
        <div>
          <b>In an emergency, call a human interpreter.</b>
          <div className="note" style={{ color: 'inherit' }}>
            This demo recognises a limited vocabulary and must not be used alone for
            diagnosis, consent or medication decisions.
          </div>
        </div>
      </div>
    </>
  )
}
