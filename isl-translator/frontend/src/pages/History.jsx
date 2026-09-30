import { useCallback, useEffect, useMemo, useState } from 'react'
import { apiUrl } from '../api.js'

const MODES = ['all', 'general', 'healthcare', 'learning']

export default function History() {
  const [items, setItems] = useState([])
  const [mode, setMode] = useState('all')
  const [query, setQuery] = useState('')

  const load = useCallback(() => {
    fetch(apiUrl(`/api/sessions?mode=${mode}`)).then((r) => r.json()).then(setItems).catch(() => setItems([]))
  }, [mode])

  useEffect(load, [load])

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase()
    return q ? items.filter((i) => (i.sentence || '').toLowerCase().includes(q)
      || i.glosses.join(' ').includes(q)) : items
  }, [items, query])

  const remove = async (id) => {
    await fetch(apiUrl(`/api/sessions/${id}`), { method: 'DELETE' })
    load()
  }

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(items, null, 1)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'signbridge-history.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <>
      <h2 className="title">Translation history</h2>
      <p className="lead">Every finished sentence from this machine.</p>

      <div className="card">
        <div className="pillrow">
          {MODES.map((m) => (
            <button key={m} className={m === mode ? 'on' : ''} onClick={() => setMode(m)}>
              {m}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
          <input type="search" placeholder="Search history…" value={query}
                 aria-label="Search history" style={{ flex: 1, minWidth: 220 }}
                 onChange={(e) => setQuery(e.target.value)} />
          <button className="btn-secondary" onClick={exportJson} disabled={!items.length}>
            Export
          </button>
        </div>

        {shown.length === 0 ? (
          <p className="note" style={{ marginTop: 16 }}>
            No sessions yet. Translate something and it will be listed here.
          </p>
        ) : (
          <table className="data" style={{ marginTop: 16 }}>
            <thead>
              <tr>
                <th>Date & time</th><th>Mode</th><th>Signs</th><th>Translation</th><th></th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => (
                <tr key={r.id}>
                  <td style={{ whiteSpace: 'nowrap' }}>{r.at}</td>
                  <td>{r.mode}</td>
                  <td>{r.glosses.join(' · ') || '—'}</td>
                  <td>{r.sentence}</td>
                  <td>
                    <button className="btn-danger" style={{ minHeight: 36, padding: '0 12px' }}
                            onClick={() => remove(r.id)} aria-label={`Delete entry from ${r.at}`}>
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  )
}
