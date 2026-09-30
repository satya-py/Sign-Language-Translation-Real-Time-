import { useEffect, useState } from 'react'
import { useApp } from '../store.jsx'
import { apiUrl } from '../api.js'

function Toggle({ on, onChange, label }) {
  return (
    <button className="toggle" role="switch" aria-checked={!!on} aria-label={label}
            onClick={() => onChange(!on)} />
  )
}

export default function Settings() {
  const { profile, settings, saveSettings, signIn } = useApp()
  const [name, setName] = useState(profile?.name || '')
  const [cameras, setCameras] = useState([])
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetch(apiUrl('/api/health')).then((r) => r.json()).then(setHealth).catch(() => {})
    navigator.mediaDevices?.enumerateDevices?.()
      .then((d) => setCameras(d.filter((x) => x.kind === 'videoinput')))
      .catch(() => {})
  }, [])

  if (!settings) return <p className="note">Loading settings…</p>

  return (
    <>
      <h2 className="title">Settings</h2>
      <p className="lead">Recognition, speech and accessibility.</p>

      <div className="two-col">
        <div>
          <div className="card">
            <h2>Profile</h2>
            <div className="field">
              <label htmlFor="pname">Display name</label>
              <input id="pname" type="text" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <button className="btn-primary" onClick={() => signIn(name, profile?.email)}>
              Save name
            </button>
            <p className="note" style={{ marginBottom: 0 }}>
              Stored only in this browser. No account, no password, nothing sent anywhere.
            </p>
          </div>

          <div className="card">
            <h2>Recognition</h2>
            <div className="label">
              Confidence threshold — {(settings.threshold * 100).toFixed(0)}%
            </div>
            <input type="range" min="0.3" max="0.9" step="0.05" value={settings.threshold}
                   aria-label="Confidence threshold"
                   onChange={(e) => saveSettings({ threshold: Number(e.target.value) })} />
            <p className="note">
              Lower catches more signs but invents more; higher is safer but drops weak signs.
              Measured best on this model: <b>35%</b> — on 166 held-out signs that reads
              93% right, 2% wrong, 5% missed, and 5% of the movement between signs
              leaking through as a word.
            </p>

            <div className="label" style={{ marginTop: 14 }}>
              Pause that completes a sentence — {settings.pause.toFixed(1)} s
            </div>
            <input type="range" min="1" max="5" step="0.5" value={settings.pause}
                   aria-label="Sentence pause"
                   onChange={(e) => saveSettings({ pause: Number(e.target.value) })} />

            <div className="switch" style={{ marginTop: 10 }}>
              <div>
                <b>Use an LLM for sentences</b>
                <div className="note">
                  Falls back to the built-in templates when no API key is set.
                </div>
              </div>
              <Toggle on={settings.use_llm} label="Use an LLM for sentences"
                      onChange={(v) => saveSettings({ use_llm: v })} />
            </div>
          </div>
        </div>

        <div>
          <div className="card">
            <h2>Accessibility</h2>
            <div className="switch">
              <span>Speak translations aloud</span>
              <Toggle on={settings.speech} label="Speak translations aloud"
                      onChange={(v) => saveSettings({ speech: v })} />
            </div>
            <div className="switch">
              <span>Large text</span>
              <Toggle on={settings.large_text} label="Large text"
                      onChange={(v) => saveSettings({ large_text: v })} />
            </div>
            <div className="switch">
              <span>High contrast</span>
              <Toggle on={settings.high_contrast} label="High contrast"
                      onChange={(v) => saveSettings({ high_contrast: v })} />
            </div>
            <div className="switch">
              <span>Reduced animation</span>
              <Toggle on={settings.reduced_motion} label="Reduced animation"
                      onChange={(v) => saveSettings({ reduced_motion: v })} />
            </div>
          </div>

          <div className="card">
            <h2>Devices</h2>
            <div className="field">
              <label htmlFor="cam">Camera</label>
              <select id="cam" style={{ minHeight: 48, width: '100%', borderRadius: 12,
                                        border: '1.5px solid var(--line-strong)', padding: '0 12px' }}>
                {cameras.length === 0 && <option>Allow camera access to list devices</option>}
                {cameras.map((c, i) => (
                  <option key={c.deviceId || i}>{c.label || `Camera ${i + 1}`}</option>
                ))}
              </select>
            </div>
            <p className="note" style={{ marginBottom: 0 }}>
              Device names appear after you have allowed camera access once.
            </p>
          </div>

          <div className="card">
            <h2>Model</h2>
            {health ? (
              <table className="data">
                <tbody>
                  <tr><td>Signs</td><td>{health.signs.length}</td></tr>
                  <tr><td>Classes (with idle)</td><td>{health.classes}</td></tr>
                  <tr><td>Validation accuracy</td>
                      <td>{(health.validation_accuracy * 100).toFixed(1)}%</td></tr>
                  <tr><td>Live-pipeline accuracy</td><td>95.4% (1185 clips)</td></tr>
                </tbody>
              </table>
            ) : <p className="note">Backend not reachable.</p>}
          </div>
        </div>
      </div>
    </>
  )
}
