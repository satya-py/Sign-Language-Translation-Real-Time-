import { useEffect, useRef, useState } from 'react'

/**
 * The camera viewport with its HUD, shared by every page that translates.
 * Bounding-box style HUD pills follow DESIGN.md: dark glass, teal when live.
 *
 * The framing row matters more than it looks: the pose tracker follows exactly one
 * person, so a second person in the room is regularly the one it follows, and every
 * sign then comes back at 20-30% confidence. Cropping to the signer costs nothing
 * and is the difference between a demo that works and one that does not.
 */
export default function CameraPanel({ videoRef, translator, label = 'Camera' }) {
  const [on, setOn] = useState(false)
  const [error, setError] = useState('')
  const [framing, setFramingState] = useState({ zoom: 1, x: 0.5 })
  const weak = useRef(0)
  const [hint, setHint] = useState('')
  const signing = translator.state === 'signing'
  const confidence = translator.last ? Math.round(translator.last.prob * 100) : null

  const frame = (next) => {
    const merged = { ...framing, ...next }
    setFramingState(merged)
    translator.setFraming?.(merged)
  }

  // Count rejected guesses in a row and say something useful about it, rather than
  // leaving the signer wondering why nothing appears.
  useEffect(() => {
    if (!translator.last) return
    if (translator.last.prob < 0.35) {
      weak.current += 1
      if (weak.current >= 4) {
        setHint('The signs are being read but not confidently. Try: one person in '
              + 'frame, head and both hands visible, and zoom in below.')
      }
    } else {
      weak.current = 0
      setHint('')
    }
  }, [translator.last])

  const start = async () => {
    try {
      await translator.startCamera()
      setOn(true)
      setError('')
    } catch (err) {
      setError(`Camera blocked: ${err.message}`)
    }
  }

  return (
    <div>
      <div className="label">{label}</div>
      <div className="stage">
        <video ref={videoRef} autoPlay playsInline muted />
        {!on && (
          <div className="placeholder">
            {error || (
              <>Press <b>Start camera</b> and sign in front of the lens.<br />
                Keep your head and both hands in view.</>
            )}
          </div>
        )}
        <div className="hud">
          <span className={`pill ${signing ? 'live' : ''}`}>
            {on ? (signing ? '● signing' : 'waiting') : 'camera off'}
          </span>
          {translator.last && (
            <span className={`pill ${confidence < 50 ? 'warn' : ''}`}>
              {translator.last.label.replace(/_/g, ' ')} {confidence}%
            </span>
          )}
          {translator.fps > 0 && <span className="pill">FPS {translator.fps.toFixed(0)}</span>}
        </div>
      </div>
      {on && (
        <div className="pillrow" style={{ marginTop: 10, alignItems: 'center' }}>
          <span className="note" style={{ margin: 0 }}>Frame the signer</span>
          {[['Everyone', 1, 0.5], ['Left', 1.8, 0.28], ['Middle', 1.8, 0.5],
            ['Right', 1.8, 0.72]].map(([name, zoom, x]) => (
            <button key={name} type="button"
                    className={framing.zoom === zoom && framing.x === x ? 'on' : ''}
                    onClick={() => frame({ zoom, x })}>{name}</button>
          ))}
          <input type="range" min="1" max="2.5" step="0.1" value={framing.zoom}
                 aria-label="Zoom on the signer"
                 onChange={(e) => frame({ zoom: Number(e.target.value) })}
                 style={{ flex: '1 1 120px', minWidth: 110 }} />
        </div>
      )}
      {hint && <p className="note" style={{ marginBottom: 0 }}>{hint}</p>}
      <div className="controls">
        {on ? (
          <button className="btn-danger" onClick={() => { translator.stopCamera(); setOn(false) }}>
            Stop camera
          </button>
        ) : (
          <button className="btn-primary" onClick={start}>Start camera</button>
        )}
        <button className="btn-secondary" onClick={translator.finish}>Finish sentence</button>
        <button className="btn-secondary" onClick={translator.clear}>Clear</button>
      </div>
    </div>
  )
}
