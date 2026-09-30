import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslator } from '../useTranslator.js'
import { useApp } from '../store.jsx'
import { apiUrl } from '../api.js'

/**
 * Teach the model your own signs.
 *
 * This is the honest fix for "it works on the dataset but not on my camera": the
 * shipped model learned from INCLUDE studio video, so a few takes from the camera
 * and the person who will demo it matter more than any tuning.
 *
 * The list comes from the sentence engine's vocabulary, not from the video
 * library, so words with no reference clip (WANT, NO, WHERE ...) are offered too —
 * those are precisely the ones the model is missing, and the grammar is already
 * ready for them.
 *
 * For those words the first take becomes the reference clip: the browser records it
 * while the keypoints are captured, and it is shown to whoever records the next
 * takes. No open dataset covers PAIN, HEAD or WHERE, and a clip of the sign this
 * model actually learned is a better reference than a clip of a different dialect.
 */
export default function Teach() {
  const videoRef = useRef(null)
  const t = useTranslator(videoRef)
  const { profile } = useApp()
  const [status, setStatus] = useState(null)
  const [word, setWord] = useState('')
  const [recording, setRecording] = useState(false)
  const [cameraOn, setCameraOn] = useState(false)
  const [message, setMessage] = useState('')
  const [training, setTraining] = useState(false)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const [trainLog, setTrainLog] = useState('')
  const [regressed, setRegressed] = useState([])

  const loadStatus = useCallback(() => {
    fetch(apiUrl('/api/teach')).then((r) => r.json()).then((s) => {
      setStatus(s)
      setWord((w) => w || s.signs.find((x) => !x.in_model)?.word || s.signs[0]?.word || '')
    }).catch(() => {})
  }, [])

  useEffect(loadStatus, [loadStatus])

  useEffect(() => {
    if (!t.taught) return
    setMessage(`Saved ${t.taught.frames} frames — ${t.taught.takes} take(s) of `
      + t.taught.word.replace(/_/g, ' '))
    loadStatus()
  }, [t.taught, loadStatus])

  useEffect(() => {
    if (!training) return undefined
    const timer = setInterval(async () => {
      const s = await (await fetch(apiUrl('/api/retrain/status'))).json()
      setTrainLog(s.log || '')
      if (!s.running) {
        setTraining(false)
        loadStatus()
        setMessage(s.last?.ok
          ? `Training finished in ${s.last.seconds}s — ${s.signs} signs. `
            + 'Open a translation page to use it.'
          : `Training failed: ${s.last?.summary}`)
        setRegressed(s.last?.regressed || [])
      }
    }, 2000)
    return () => clearInterval(timer)
  }, [training, loadStatus])

  const groups = useMemo(() => {
    if (!status) return []
    const missing = status.signs.filter((s) => !s.in_model)
    const known = status.signs.filter((s) => s.in_model)
    const byTier = {}
    missing.forEach((s) => { (byTier[s.tier] ||= []).push(s) })
    const out = Object.keys(byTier).sort().map((tier) => ({
      title: `Not recognised yet — ${status.tiers[tier] || 'other'}`,
      signs: byTier[tier],
    }))
    if (known.length) {
      out.push({ title: 'Already recognised — extra takes improve accuracy', signs: known })
    }
    return out
  }, [status])

  const current = status?.signs.find((s) => s.word === word)
  const taken = current?.takes || 0
  const pretty = word.replace(/_/g, ' ')

  const startCamera = async () => {
    try {
      await t.startCamera()
      setCameraOn(true)
    } catch (err) {
      setMessage(`Camera blocked: ${err.message}`)
    }
  }

  // Record the take in the browser too, but only when this sign has no reference
  // clip — then the take itself becomes the reference.
  const startRecorder = (forWord) => {
    const stream = videoRef.current?.srcObject
    if (!stream || typeof MediaRecorder === 'undefined') return
    const type = ['video/webm;codecs=vp9', 'video/webm', 'video/mp4']
      .find((m) => MediaRecorder.isTypeSupported?.(m))
    if (!type) return
    try {
      const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 800000 })
      chunksRef.current = []
      rec.ondataavailable = (e) => { if (e.data.size) chunksRef.current.push(e.data) }
      rec.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type })
        chunksRef.current = []
        if (blob.size < 2048) return
        const body = new FormData()
        body.append('word', forWord)
        body.append('file', blob, `take.${type.includes('webm') ? 'webm' : 'mp4'}`)
        try {
          const r = await fetch(apiUrl('/api/teach/video'), { method: 'POST', body })
          if (r.ok) loadStatus()
        } catch { /* the keypoints are saved either way — that is what trains */ }
      }
      rec.start(250)
      recorderRef.current = rec
    } catch { /* ignore: recording the reference is a bonus, not the point */ }
  }

  const toggleRecord = () => {
    if (!cameraOn || !word) return
    if (recording) {
      t.send({ cmd: 'teach_stop', signer: profile?.name || 'web' })
      setRecording(false)
      if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
      recorderRef.current = null
    } else {
      setMessage('')
      t.send({ cmd: 'teach_start', word })
      setRecording(true)
      if (!current?.has_video) startRecorder(word)
    }
  }

  const dropOwnClip = async () => {
    await fetch(apiUrl(`/api/teach/video/${word}`), { method: 'DELETE' })
    setMessage(`Reference take for ${pretty} removed — the next take replaces it.`)
    loadStatus()
  }

  const retrain = async () => {
    setTraining(true)
    setRegressed([])
    setMessage('Training… about five minutes. You can watch the log below.')
    await fetch(apiUrl('/api/retrain'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
  }

  // Teaching a new sign can cost accuracy on the ones already taught. The backend
  // keeps the model it replaced, so that is recoverable rather than a crisis.
  const revert = async () => {
    const r = await fetch(apiUrl('/api/retrain/revert'), { method: 'POST' })
    const body = await r.json()
    setMessage(r.ok ? `Back on the previous model — ${body.signs} signs.`
                    : `Could not go back: ${body.detail}`)
    setRegressed([])
    loadStatus()
  }

  return (
    <>
      <h2 className="title">Teach SignBridge your signs</h2>
      <p className="lead">
        Record a few takes from this camera, then retrain. The model learns your camera,
        your lighting and your signing — and learns signs it has never seen.
      </p>

      {status && (
        <div className="model-strip" style={{ marginBottom: 'var(--gutter)' }}>
          <span className="dot" aria-hidden="true" />
          <div>
            <b>{status.totals.in_model} signs recognised · {status.totals.missing} waiting to be taught</b>
            <div className="note" style={{ margin: 0 }}>
              {status.totals.takes} takes recorded so far
            </div>
          </div>
        </div>
      )}

      <div className="split">
        <section>
          <div className="card">
            <div className="label">1 · Pick a sign</div>
            <select value={word} onChange={(e) => setWord(e.target.value)}
                    aria-label="Sign to teach"
                    style={{ minHeight: 48, width: '100%', borderRadius: 12,
                             border: '1.5px solid var(--line-strong)', padding: '0 12px',
                             fontSize: 16 }}>
              {groups.map((group) => (
                <optgroup key={group.title} label={group.title}>
                  {group.signs.map((s) => (
                    <option key={s.word} value={s.word}>
                      {s.word.replace(/_/g, ' ')} ({s.takes} takes)
                      {s.has_video ? (s.own_video ? ' · your take' : '') : ' · needs a reference'}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>

            {current && (
              <div className="pillrow" style={{ marginTop: 12 }}>
                <button type="button">{current.kind}</button>
                <button type="button" className={current.in_model ? 'on' : ''}>
                  {current.in_model ? 'in the model' : 'not recognised yet'}
                </button>
                <button type="button">{taken} takes</button>
              </div>
            )}

            {current?.has_video ? (
              <>
                <div className="label" style={{ marginTop: 16 }}>
                  2 · Copy this{current.own_video ? ' — your own reference take' : ''}
                </div>
                <div className="stage" style={{ aspectRatio: '16 / 9' }}>
                  <video key={word} src={apiUrl(`/api/clip/${word}`)} autoPlay muted loop
                         playsInline controls />
                </div>
                {current.own_video && (
                  <p className="note">
                    This is the take you recorded for <b>{pretty}</b>. Everyone who records
                    more takes copies it, so the model stays consistent.{' '}
                    <button className="btn-ghost" style={{ minHeight: 32 }}
                            onClick={dropOwnClip}>Record a new reference</button>
                  </p>
                )}
              </>
            ) : (
              <>
                <div className="label" style={{ marginTop: 16 }}>2 · Look up the sign</div>
                <p className="note">
                  No open dataset covers <b>{pretty}</b>. Check the sign first, then record
                  it the same way every time — the model only learns what you show it.
                  <b> Your first take is kept as the reference clip</b>, so the next person
                  copies exactly what you signed.
                </p>
                <div className="pillrow">
                  <a href={`https://indiansignlanguage.org/?s=${encodeURIComponent(pretty)}`}
                     target="_blank" rel="noreferrer">
                    <button className="btn-secondary" style={{ minHeight: 40 }}>
                      ISL dictionary
                    </button>
                  </a>
                  <a href={`https://www.youtube.com/results?search_query=${encodeURIComponent('Indian sign language ' + pretty)}`}
                     target="_blank" rel="noreferrer">
                    <button className="btn-secondary" style={{ minHeight: 40 }}>
                      Video search
                    </button>
                  </a>
                </div>
              </>
            )}
          </div>
        </section>

        <section>
          <div className="card">
            <div className="label">3 · Record {status?.recommended || 5}+ takes of “{pretty}”</div>
            <div className="stage" style={{ aspectRatio: '4 / 3' }}>
              <video ref={videoRef} autoPlay playsInline muted />
              {!cameraOn && <div className="placeholder">Start the camera to record takes.</div>}
              <div className="hud">
                <span className={`pill ${recording ? 'live' : ''}`}>
                  {recording ? `● recording ${t.teachFrames || 0} frames` : 'ready'}
                </span>
                <span className="pill">{taken}/{status?.recommended || 5} takes</span>
              </div>
            </div>
            <div className="controls">
              {cameraOn ? (
                <button className={recording ? 'btn-danger' : 'btn-primary'} onClick={toggleRecord}>
                  {recording ? 'Stop take' : 'Record a take'}
                </button>
              ) : (
                <button className="btn-primary" onClick={startCamera}>Start camera</button>
              )}
              <button className="btn-secondary" onClick={retrain}
                      disabled={training || !status?.totals.takes}>
                {training ? 'Training…' : `Retrain (${status?.totals.takes || 0} takes)`}
              </button>
            </div>
            <div className="bar" style={{ marginTop: 12 }}>
              <span style={{ width: `${Math.min(100, (taken / (status?.recommended || 5)) * 100)}%` }} />
            </div>
            {message && <p className="note" style={{ marginBottom: 0 }}>{message}</p>}
            {regressed.length > 0 && (
              <div className="note" style={{ marginBottom: 0 }}>
                <b>These signs got worse in this training run:</b>
                <ul style={{ margin: '6px 0 8px 18px' }}>
                  {regressed.map((line) => <li key={line}>{line}</li>)}
                </ul>
                Record a few more takes of them, or go back to the model from before.
                <div className="pillrow" style={{ marginTop: 8 }}>
                  <button className="btn-secondary" style={{ minHeight: 36 }}
                          onClick={revert}>Use the previous model</button>
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <div className="label">How to get a model that works on stage</div>
            <ol style={{ paddingLeft: 20, lineHeight: '26px', margin: 0 }}>
              <li>Record <b>5 or more takes</b> per sign, from the camera you will demo with.</li>
              <li>Vary a little: sit closer and further, turn slightly, sign faster and slower.</li>
              <li>Ask a second person to record takes too — that is what makes it work for
                  someone new.</li>
              <li>Teach a whole group of signs, then press <b>Retrain</b> once and wait
                  about five minutes. Do not press it again while it runs.</li>
            </ol>
            {trainLog && (
              <pre className="note" style={{ whiteSpace: 'pre-wrap', maxHeight: 160,
                                             overflow: 'auto', marginBottom: 0 }}>
                {trainLog}
              </pre>
            )}
            {status?.last && !training && (
              <p className="note" style={{ marginBottom: 0 }}>
                Last training: {status.last.at} — {status.last.ok ? 'succeeded' : 'failed'}
                {status.last.summary ? ` (${status.last.summary})` : ''}
              </p>
            )}
          </div>
        </section>
      </div>
    </>
  )
}
