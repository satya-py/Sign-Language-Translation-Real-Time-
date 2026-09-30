import { useEffect, useRef, useState } from 'react'
import { speak } from '../speech.js'
import { recordSession } from '../store.jsx'
import { useTranslator } from '../useTranslator.js'
import { apiUrl } from '../api.js'

const USES = [
  ['My name is', 'name'],
  ['I live in', 'place'],
  ['My medicine is', 'medicine'],
  ['Just the word', 'plain'],
]

/**
 * Fingerspelling: names, places and anything the word model was never taught.
 *
 * A letter is a held shape rather than a movement, so the backend accepts a letter
 * only after it has been steady for several frames, and a repeated letter needs a
 * break in between — otherwise "ANNA" comes out as "AAAAAANNNNNA".
 */
export default function Fingerspell() {
  const videoRef = useRef(null)
  const t = useTranslator(videoRef)
  const [available, setAvailable] = useState(null)

  // Switching model frees the one that was loaded: each carries its own
  // MediaPipe graph, and two of them do not fit comfortably on this machine.
  const chooseEngine = async (engine) => {
    const answer = await fetch(apiUrl('/api/fingerspell'), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ engine }),
    })
    if (answer.ok) {
      const status = await (await fetch(apiUrl('/api/fingerspell'))).json()
      setAvailable(status)
    }
  }
  const [cameraOn, setCameraOn] = useState(false)
  const [spelling, setSpelling] = useState(false)
  const [use, setUse] = useState('name')
  const [words, setWords] = useState([])
  const [message, setMessage] = useState('')

  useEffect(() => {
    fetch(apiUrl('/api/fingerspell')).then((r) => r.json()).then(setAvailable).catch(() => setAvailable(false))
  }, [])

  // The backend answers spell_stop with the finished word.
  useEffect(() => {
    if (!t.spelled) return
    const word = t.spelled.word
    setSpelling(false)
    if (!word) { setMessage('Nothing was spelled.'); return }
    setWords((prev) => [...prev, word])
    const sentence = buildSentence(use, word)
    setMessage(sentence)
    speak(sentence)
    recordSession({ mode: 'general', sentence, glosses: t.spelled.letters, source: 'fingerspell' })
  }, [t.spelled])                     // eslint-disable-line react-hooks/exhaustive-deps

  const buildSentence = (kind, word) => {
    const pretty = word.charAt(0) + word.slice(1).toLowerCase()
    if (kind === 'name') return `My name is ${pretty}.`
    if (kind === 'place') return `I live in ${pretty}.`
    if (kind === 'medicine') return `My medicine is ${pretty}.`
    return pretty
  }

  const start = async () => {
    try {
      await t.startCamera()
      setCameraOn(true)
    } catch (err) {
      setMessage(`Camera blocked: ${err.message}`)
    }
  }

  const toggleSpelling = () => {
    if (spelling) {
      t.send({ cmd: 'spell_stop' })
    } else {
      setMessage('')
      t.send({ cmd: 'spell_start' })
      setSpelling(true)
    }
  }

  const current = t.spellCurrent
  const letters = t.spellLetters || []

  if (available && available.available === false) {
    return (
      <>
        <h2 className="title">Fingerspelling</h2>
        <p className="lead">Spell names and words that have no sign of their own.</p>
        <div className="card">
          <p className="note" style={{ marginBottom: 0 }}>
            No letter model yet. Record letters with{' '}
            <code>fingerspell/record_letters.py</code> and run{' '}
            <code>fingerspell/train_letters.py</code>.
          </p>
        </div>
      </>
    )
  }

  return (
    <>
      <h2 className="title">Fingerspelling</h2>
      <p className="lead">
        For names, places and anything the word model was never taught — spell it letter
        by letter.
      </p>

      <div className="split">
        <section>
          <div className="card">
            <div className="label">Camera</div>
            <div className="stage">
              <video ref={videoRef} autoPlay playsInline muted />
              {!cameraOn && (
                <div className="placeholder">
                  Start the camera, press <b>Start spelling</b>, then hold each letter
                  still for about half a second.
                </div>
              )}
              <div className="hud">
                <span className={`pill ${spelling ? 'live' : ''}`}>
                  {spelling ? '● spelling' : 'idle'}
                </span>
                {current && (
                  <span className={`pill ${current.prob < 0.6 ? 'warn' : ''}`}>
                    {current.label} {Math.round(current.prob * 100)}%
                  </span>
                )}
              </div>
            </div>

            <div className="controls">
              {cameraOn ? (
                <button className="btn-danger" onClick={() => { t.stopCamera(); setCameraOn(false) }}>
                  Stop camera
                </button>
              ) : (
                <button className="btn-primary" onClick={start}>Start camera</button>
              )}
              <button className={spelling ? 'btn-danger' : 'btn-primary'}
                      onClick={toggleSpelling} disabled={!cameraOn}>
                {spelling ? 'Finish word' : 'Start spelling'}
              </button>
              <button className="btn-secondary" disabled={!spelling}
                      onClick={() => t.send({ cmd: 'spell_backspace' })}>
                Delete last letter
              </button>
            </div>

            <p className="note" style={{ marginBottom: 0 }}>
              To spell a double letter (like the two N&rsquo;s in ANNA), drop your hand for a
              moment between them — that break is what separates them.
            </p>
          </div>
        </section>

        <section>
          <div className="card" aria-live="polite">
            <div className="label">Letters</div>
            <div className="glosses">
              {letters.length === 0 && <span className="note">Nothing spelled yet.</span>}
              {letters.map((l, i) => <span className="gloss" key={`${l}-${i}`}>{l}</span>)}
            </div>

            <div className="label" style={{ marginTop: 20 }}>Word</div>
            <div className="sentence">{t.spellWord || ''}</div>

            <div className="label" style={{ marginTop: 14 }}>Use it as</div>
            <div className="pillrow">
              {USES.map(([text, key]) => (
                <button key={key} className={key === use ? 'on' : ''} onClick={() => setUse(key)}>
                  {text}
                </button>
              ))}
            </div>

            {message && <div className="reply-out">{message}</div>}

            <div className="meta">
              <span className={`chip ${t.connection === 'connected' ? 'live' : 'err'}`}>
                {t.connection}
              </span>
              {available?.validation_accuracy && (
                <span className="chip">
                  {Math.round(available.validation_accuracy * 100)}% on held-out frames
                </span>
              )}
            </div>

            {available?.engines?.length > 1 && (
              <div className="card" style={{ marginTop: 12 }}>
                <div className="label">Which letter model</div>
                <div className="pillrow">
                  {available.engines.map((engine) => (
                    <button key={engine.id}
                            className={engine.id === available.engine ? 'on' : ''}
                            onClick={() => chooseEngine(engine.id)}>
                      {engine.name}
                    </button>
                  ))}
                </div>
                {available.engines.map((engine) => (
                  <p className="note" key={engine.id} style={{ marginBottom: 2 }}>
                    <b>{engine.name}</b> — {engine.count} letters;{' '}
                    {engine.accuracy
                      ? `${Math.round(engine.accuracy * 100)}% measured`
                      : 'accuracy not measured'}. {engine.measured_on}
                  </p>
                ))}
              </div>
            )}
          </div>

          {words.length > 0 && (
            <div className="card">
              <div className="label">Spelled this session</div>
              <div className="pillrow">
                {words.map((w, i) => (
                  <button key={`${w}-${i}`} onClick={() => speak(buildSentence(use, w))}>{w}</button>
                ))}
              </div>
            </div>
          )}

          <div className="card">
            <div className="label">Honest limits</div>
            <p className="note" style={{ marginBottom: 0 }}>
              The letter model was trained on one signer&rsquo;s recordings in a single
              session, so it is strongest for that person and that camera. Teach it more
              frames from other people to make it general. Letters that look alike
              (H and K, L and X, Z and C) are the ones it confuses.
            </p>
          </div>
        </section>
      </div>
    </>
  )
}
