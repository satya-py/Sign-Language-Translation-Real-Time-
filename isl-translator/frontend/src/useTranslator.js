import { useCallback, useEffect, useRef, useState } from 'react'
import { speak } from './speech.js'
import { wsUrl } from './api.js'

/**
 * Owns the WebSocket to the backend and the camera frame pump.
 *
 * Frames are captured at 640x480: MediaPipe needs that much detail to find
 * fingers, and smaller frames measurably hurt recognition.
 */
export function useTranslator(videoRef) {
  const ws = useRef(null)
  const sending = useRef(false)
  const canvas = useRef(null)
  const [connection, setConnection] = useState('connecting')
  const [state, setState] = useState({
    glosses: [], sentence: '', source: '', state: 'waiting', last: null, fps: 0,
    lastGlosses: [],          // the signs that produced the current sentence
    // The sentence on screen now grows with every sign, so a separate field marks
    // the one that is actually finished - only that belongs in the transcript.
    finalSentence: '', finalAt: 0, building: false,
  })
  const [vocab, setVocab] = useState([])
  const [taught, setTaught] = useState(null)          // answer to a finished take
  const [teachFrames, setTeachFrames] = useState(0)
  const [spell, setSpell] = useState({ letters: [], word: '', current: null })
  const [spelled, setSpelled] = useState(null)        // answer to a finished word

  useEffect(() => {
    let closed = false
    const connect = () => {
      const socket = new WebSocket(wsUrl('/ws'))
      ws.current = socket
      socket.onopen = () => setConnection('connected')
      socket.onclose = () => {
        if (closed) return
        setConnection('reconnecting')
        setTimeout(connect, 1500)
      }
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if (data.type === 'hello') { setVocab(data.vocab); return }
        if (data.type === 'teaching') { setTeachFrames(data.frames); return }
        if (data.type === 'taught') { setTaught(data); setTeachFrames(0); return }
        if (data.type === 'spelling') {
          setSpell({ letters: data.letters || [], word: data.word || '', current: data.current })
          return
        }
        if (data.type === 'spelled') {
          setSpelled(data)
          setSpell({ letters: [], word: '', current: null })
          return
        }
        if (data.type === 'spell_unavailable') { setSpelled({ word: '', letters: [] }); return }
        setState((prev) => ({
          ...prev,
          ...data,
          // The backend clears its gloss list when a sentence completes, so keep
          // the words that produced it for the transcript and the history.
          lastGlosses: data.said?.length ? data.said
            : (data.speak && prev.glosses.length ? prev.glosses : prev.lastGlosses),
          building: data.source === 'building',
          finalSentence: data.speak && data.sentence ? data.sentence : prev.finalSentence,
          finalAt: data.speak && data.sentence ? Date.now() : prev.finalAt,
        }))
        if (data.speak && data.sentence) speak(data.sentence)
      }
    }
    connect()
    return () => { closed = true; ws.current?.close() }
  }, [])

  const send = useCallback((payload) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(payload))
    }
  }, [])

  // Which part of the camera image to send. MediaPipe Holistic follows exactly one
  // person; with somebody else in the room it regularly follows them instead, and
  // the signs come back at 20-30% confidence. Cropping to the signer fixes that
  // without touching the model.
  const framing = useRef({ zoom: 1, x: 0.5 })

  const setFraming = useCallback((next) => {
    framing.current = { ...framing.current, ...next }
  }, [])

  const pump = useCallback(() => {
    if (!sending.current) return
    const video = videoRef.current
    if (video && video.readyState >= 2 && ws.current?.readyState === WebSocket.OPEN) {
      if (!canvas.current) canvas.current = document.createElement('canvas')
      const c = canvas.current
      c.width = 640
      c.height = 480
      const vw = video.videoWidth || 640
      const vh = video.videoHeight || 480
      const { zoom, x } = framing.current
      const sw = vw / zoom
      const sh = vh / zoom
      const sx = Math.max(0, Math.min(vw - sw, x * vw - sw / 2))
      const sy = Math.max(0, Math.min(vh - sh, vh / 2 - sh / 2))
      c.getContext('2d').drawImage(video, sx, sy, sw, sh, 0, 0, c.width, c.height)
      send({ frame: c.toDataURL('image/jpeg', 0.7) })
    }
    setTimeout(pump, 70)             // about 14 frames per second
  }, [send, videoRef])

  const startCamera = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    })
    videoRef.current.srcObject = stream
    sending.current = true
    pump()
  }, [pump, videoRef])

  const stopCamera = useCallback(() => {
    sending.current = false
    const stream = videoRef.current?.srcObject
    stream?.getTracks().forEach((t) => t.stop())
    if (videoRef.current) videoRef.current.srcObject = null
  }, [videoRef])

  useEffect(() => stopCamera, [stopCamera])

  return {
    ...state,
    vocab,
    taught,
    teachFrames,
    spellLetters: spell.letters,
    spellWord: spell.word,
    spellCurrent: spell.current,
    spelled,
    send,
    setFraming,
    connection,
    startCamera,
    stopCamera,
    clear: () => { send({ cmd: 'clear' }); setState((p) => ({ ...p, glosses: [], sentence: '' })) },
    finish: () => send({ cmd: 'finish' }),
    setFromUpload: (data) => setState((p) => ({ ...p, ...data, lastGlosses: data.glosses || [] })),
  }
}
