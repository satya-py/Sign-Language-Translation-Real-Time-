import { useEffect, useMemo, useRef, useState } from 'react'
import { apiUrl } from '../api.js'

const PHRASES = [
  ['HELLO → HOW ARE YOU', 'Hello, how are you?'],
  ['I → SICK → TODAY', 'I am sick today.'],
  ['I → HOT → NIGHT', 'I have a fever at night.'],
  ['I → WEAK → MORNING', 'I am weak in the morning.'],
  ['I → COLD → NIGHT', 'I am feeling cold at night.'],
  ['I → MEDICINE', 'I need medicine.'],
  ['I → BATHROOM', 'I am going to the bathroom.'],
  ['MOTHER → SICK → YESTERDAY', 'My mother was sick yesterday.'],
  ['PATIENT → BED → WEAK', 'The patient is in bed and weak.'],
  ['THANK YOU', 'Thank you.'],
]

// Grouping only for browsing; the model treats every sign the same way.
const CATEGORIES = {
  Greetings: ['hello', 'how_are_you', 'thank_you'],
  Healthcare: ['doctor', 'patient', 'hospital', 'medicine', 'sick', 'healthy', 'hot', 'cold',
    'weak', 'strong', 'bed', 'bathroom', 'dead', 'alive'],
  People: ['i', 'you', 'he', 'she', 'mother', 'father', 'family', 'friend', 'old', 'young'],
  Time: ['today', 'tomorrow', 'yesterday', 'morning', 'night', 'time'],
  Everyday: ['house', 'school', 'money', 'good', 'bad'],
}

const STEPS = [
  'Sit so your head and both hands are inside the frame.',
  'Watch the clip once, then copy the hand shape and the movement.',
  'Drop your hands and hold still for about a second — that ends the sign.',
  'Sign the next word, then wait two seconds to finish the sentence.',
]

function SignTile({ item, onPick }) {
  const [playing, setPlaying] = useState(false)
  const videoRef = useRef(null)

  useEffect(() => {
    if (playing) videoRef.current?.play().catch(() => { /* controls remain */ })
  }, [playing])

  return (
    <div className="tile">
      <button className={`frame${playing ? ' playing' : ''}`}
              onClick={() => { setPlaying(true); onPick?.(item.word) }}
              aria-label={`Play the sign for ${item.word.replace(/_/g, ' ')}`}>
        {playing ? (
          <video ref={videoRef} src={apiUrl(`/api/clip/${item.word}`)} muted loop playsInline controls
                 preload="auto" />
        ) : (
          <img src={apiUrl(`/api/poster/${item.word}`)} alt="" loading="lazy" />
        )}
      </button>
      <div className="word">{item.word.toUpperCase().replace(/_/g, ' ')}</div>
      <div className="sub">
        {item.seconds}s · tap to play
        {item.in_model ? '' : ' · not recognised yet — teach it'}
      </div>
    </div>
  )
}

export default function Learn() {
  const [items, setItems] = useState([])
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('All')

  useEffect(() => {
    fetch(apiUrl('/api/gallery')).then((r) => r.json()).then(setItems).catch(() => setItems([]))
  }, [])

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase().replace(/\s+/g, '_')
    const inCategory = category === 'All' ? items
      : items.filter((i) => (CATEGORIES[category] || []).includes(i.word))
    return q ? inCategory.filter((i) => i.word.includes(q)) : inCategory
  }, [items, query, category])

  return (
    <>
      <h2 className="title">Learn Indian Sign Language</h2>
      <p className="lead">Practise the signs this system understands, then try them live.</p>

      <div className="card">
        <h2>How to perform a sign</h2>
        <ol style={{ paddingLeft: 20, lineHeight: '28px', margin: '0 0 8px' }}>
          {STEPS.map((s) => <li key={s}>{s}</li>)}
        </ol>
        <p className="note" style={{ marginBottom: 0 }}>
          ISL has no separate sign for several clinical words, so the nearest sign is read in
          context: <b>HOT = fever</b>, <b>COLD = chills</b>, <b>BAD = not feeling well</b>.
        </p>
      </div>

      <div className="card">
        <h2>Sentences to try</h2>
        <table className="phrases">
          <tbody>
            {PHRASES.map(([signs, english]) => (
              <tr key={signs}><td className="sign">{signs}</td><td>{english}</td></tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>Sign library — {items.length} signs</h2>
        <div className="pillrow" style={{ marginBottom: 12 }}>
          {['All', ...Object.keys(CATEGORIES)].map((c) => (
            <button key={c} className={c === category ? 'on' : ''} onClick={() => setCategory(c)}>
              {c}
            </button>
          ))}
        </div>
        <input type="search" placeholder="Search a sign…" value={query}
               aria-label="Search a sign" style={{ maxWidth: 360, marginBottom: 16 }}
               onChange={(e) => setQuery(e.target.value)} />
        {shown.length === 0 && items.length > 0 && (
          <p className="note">No sign matches that search.</p>
        )}
        <div className="grid">
          {shown.map((item) => <SignTile key={item.word} item={item} />)}
        </div>
      </div>
    </>
  )
}
