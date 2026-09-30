import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiUrl } from '../api.js'

const FEATURES = [
  ['◉', '', 'Real-time translation',
    'Sign in front of any webcam and read the English sentence as it is built.'],
  ['✚', 'teal', 'Healthcare communication',
    'Made for the moment a deaf patient meets a doctor without an interpreter.'],
  ['🎓', 'indigo', 'Learn Indian Sign Language',
    'Every sign the system knows, demonstrated by deaf signers, ready to copy.'],
]

const STEPS = [
  ['Camera', 'Your webcam captures the signing.'],
  ['Keypoints', 'MediaPipe finds the body and both hands, 22 times a second.'],
  ['Recognition', 'A small model names each sign and knows when nobody is signing.'],
  ['Sentence', 'ISL word order becomes an English sentence, spoken aloud.'],
]

export default function Landing() {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetch(apiUrl('/api/health')).then((r) => r.json()).then(setHealth).catch(() => {})
  }, [])

  return (
    <main>
      <section className="hero-wrap" style={{ padding: '18px 0 36px' }}>
        <div>
          <h1 className="hero-title">Breaking communication barriers with AI</h1>
          <p className="hero-sub">
            Real-time Indian Sign Language translation that helps deaf patients and doctors
            understand each other, with no interpreter in the room.
          </p>
          <div className="hero-actions">
            <Link to="/app/live"><button className="btn-primary">Start translating</button></Link>
            <Link to="/app/learn"><button className="btn-secondary">Explore sign library</button></Link>
          </div>
          <p className="note" style={{ marginTop: 18 }}>
            {health
              ? `${health.signs.length} signs live · 95% accuracy measured on 1185 clips · ~0.3 s delay`
              : 'Connecting to the recognition service…'}
          </p>
        </div>
        <div className="hero-art">
          <img src={apiUrl("/api/poster/how_are_you")} alt="A signer performing an Indian Sign Language sign" />
          <div className="hero-badge" style={{ top: 16, left: 16 }}>● ISL detected</div>
          <div className="hero-badge" style={{ bottom: 16, left: 16 }}>Hello, how are you?</div>
          <div className="hero-badge" style={{ bottom: 16, right: 16 }}>Confidence 96%</div>
        </div>
      </section>

      <section className="feature-grid">
        {FEATURES.map(([icon, tone, title, text]) => (
          <div className="card feature" key={title}>
            <div className={`icon ${tone}`} aria-hidden="true">{icon}</div>
            <h3>{title}</h3>
            <p className="note" style={{ marginBottom: 0 }}>{text}</p>
          </div>
        ))}
      </section>

      <section className="card" style={{ marginTop: 'var(--gutter)' }}>
        <h2>How it works</h2>
        <div className="steps">
          {STEPS.map(([title, text], i) => (
            <div className="step" key={title}>
              <div className="num">{i + 1}</div>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700 }}>{title}</div>
              <p className="note">{text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="tiles" style={{ marginTop: 'var(--gutter)' }}>
        <div className="card stat">
          <div className="big">{health ? health.signs.length : '—'}</div>
          <div className="cap">signs recognised</div>
        </div>
        <div className="card stat">
          <div className="big">95%</div>
          <div className="cap">accuracy through the live pipeline</div>
        </div>
        <div className="card stat">
          <div className="big">0.31 s</div>
          <div className="cap">from end of sign to text</div>
        </div>
        <div className="card stat">
          <div className="big">22 fps</div>
          <div className="cap">on a laptop, no cloud GPU</div>
        </div>
      </section>
    </main>
  )
}
