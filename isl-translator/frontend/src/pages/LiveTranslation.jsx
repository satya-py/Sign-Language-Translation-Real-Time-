import { useEffect, useRef, useState } from 'react'
import CameraPanel from '../components/CameraPanel.jsx'
import { speak } from '../speech.js'
import { recordSession, useApp } from '../store.jsx'
import { useTranslator } from '../useTranslator.js'
import { apiUrl } from '../api.js'

/** Live ISL translation with a running transcript of finished sentences. */
export default function LiveTranslation({ mode = 'general', title = 'Live ISL translation',
                                          lead = 'Sign in front of the camera; the English sentence appears as you go.' }) {
  const videoRef = useRef(null)
  const t = useTranslator(videoRef)
  const { settings } = useApp()
  const [transcript, setTranscript] = useState([])
  const lastSaved = useRef('')
  const [fileNote, setFileNote] = useState('')

  // Only a FINISHED sentence goes into the transcript. The sentence on screen is
  // rebuilt after every sign so the signer sees it grow, and saving those would
  // fill the transcript with half sentences.
  useEffect(() => {
    if (!t.finalSentence || !t.finalAt) return
    const key = `${t.finalAt}:${t.finalSentence}`
    if (key === lastSaved.current) return
    lastSaved.current = key
    const entry = { at: new Date().toLocaleTimeString(), sentence: t.finalSentence,
                    glosses: t.lastGlosses }
    setTranscript((prev) => [...prev, entry])
    recordSession({ mode, sentence: t.finalSentence, glosses: t.lastGlosses, source: t.source })
  }, [t.finalSentence, t.finalAt, t.lastGlosses, t.source, mode])

  const upload = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    setFileNote(`Processing ${file.name}…`)
    const body = new FormData()
    body.append('file', file)
    const data = await (await fetch(apiUrl('/api/video'), { method: 'POST', body })).json()
    t.setFromUpload({ glosses: data.glosses, sentence: data.sentence, source: data.source })
    setFileNote(`${data.segments.length} segments analysed.`)
    if (data.sentence) {
      setTranscript((prev) => [...prev, { at: new Date().toLocaleTimeString(),
                                          sentence: data.sentence, glosses: data.glosses }])
      recordSession({ mode, sentence: data.sentence, glosses: data.glosses, source: data.source })
      if (settings?.speech !== false) speak(data.sentence)
    }
  }

  return (
    <>
      <h2 className="title">{title}</h2>
      <p className="lead">{lead}</p>

      <div className="split">
        <section>
          <div className="card">
            <CameraPanel videoRef={videoRef} translator={t} label="Camera" />
            <p className="note" style={{ marginBottom: 0 }}>
              After each sign, drop your hands for about a second. Two seconds of stillness
              completes the sentence.
            </p>
          </div>

          <div className="card">
            <div className="label">Conversation transcript</div>
            <div className="chat">
              {transcript.length === 0 && (
                <p className="note">Nothing yet — your finished sentences will appear here.</p>
              )}
              {transcript.map((m, i) => (
                <div className="msg patient" key={i}>
                  <span className="who" aria-hidden="true">ISL</span>
                  <div className="bubble">
                    {m.sentence}
                    <span className="time">{m.at} · {m.glosses?.join(' · ')}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section>
          <div className="card" aria-live="polite">
            <div className="label">Detected signs</div>
            <div className="glosses">
              {t.glosses.map((g, i) => (
                <span className="gloss" key={`${g}-${i}`}>{g.toUpperCase().replace(/_/g, ' ')}</span>
              ))}
            </div>

            <div className="label" style={{ marginTop: 20 }}>Current sentence</div>
            <div className="sentence">{t.sentence}</div>

            {t.last && (
              <>
                <div className="label" style={{ marginTop: 16 }}>
                  Confidence — {Math.round(t.last.prob * 100)}%
                </div>
                <div className="bar"><span style={{ width: `${t.last.prob * 100}%` }} /></div>
              </>
            )}

            <div className="meta">
              <span className={`chip ${t.state === 'signing' ? 'live' : ''}`}>{t.state}</span>
              {t.source && (
                <span className="chip">
                  {t.building ? 'sentence still growing' : `sentence by ${t.source}`}
                </span>
              )}
              <span className={`chip ${t.connection === 'connected' ? 'live' : 'err'}`}>{t.connection}</span>
            </div>

            <div className="controls">
              <button className="btn-secondary" onClick={() => speak(t.sentence)}>Speak</button>
              <button className="btn-secondary"
                      onClick={() => navigator.clipboard?.writeText(t.sentence)}>Copy</button>
            </div>
          </div>

          <div className="card">
            <div className="label">No camera? Use a recorded clip</div>
            <input type="file" accept="video/*" onChange={upload}
                   aria-label="Upload a sign language video" />
            {fileNote && <p className="note" style={{ marginBottom: 0 }}>{fileNote}</p>}
          </div>

          <div className="card">
            <div className="label">Signs this model knows ({t.vocab.length})</div>
            <p className="vocab">
              {t.vocab.length ? t.vocab.map((w) => w.replace(/_/g, ' ')).join(' · ') : 'loading…'}
            </p>
          </div>
        </section>
      </div>
    </>
  )
}
