import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useApp } from '../store.jsx'

/**
 * Login / Sign up.
 *
 * Deliberately a DEMO PROFILE: it asks for a name only, keeps it in this browser,
 * and never asks for or transmits a password. A hackathon demo has no safe place to
 * store real credentials, and pretending otherwise would be worse than being clear.
 */
function AuthPage({ mode }) {
  const isSignup = mode === 'signup'
  const { signIn } = useApp()
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')

  const submit = (event) => {
    event.preventDefault()
    signIn(name.trim() || 'Guest', email.trim())
    navigate('/app/dashboard')
  }

  return (
    <div className="auth">
      <div className="auth-form">
        <form className="inner" onSubmit={submit}>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 32, margin: '0 0 6px' }}>
            {isSignup ? 'Create your SignBridge profile' : 'Welcome back'}
          </h2>
          <p className="note" style={{ marginTop: 0 }}>
            {isSignup
              ? 'Start breaking communication barriers.'
              : 'Sign in to continue your session.'}
          </p>

          <div className="demo-banner">
            <b>Demo profile.</b> This build asks for a name only. No password is requested,
            nothing is sent to a server, and the name stays in this browser. It is used to
            greet you and to label your session history.
          </div>

          <div className="field">
            <label htmlFor="name">Your name</label>
            <input id="name" type="text" value={name} autoComplete="name"
                   onChange={(e) => setName(e.target.value)} placeholder="e.g. Satyabrata" />
          </div>
          <div className="field">
            <label htmlFor="email">Email (optional, stays on this device)</label>
            <input id="email" type="text" value={email} autoComplete="off"
                   onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
          </div>

          <button className="btn-primary" type="submit" style={{ width: '100%' }}>
            {isSignup ? 'Create profile' : 'Continue'}
          </button>

          <p className="note" style={{ marginTop: 16 }}>
            {isSignup
              ? <>Already have a profile here? <Link to="/login">Open it</Link></>
              : <>No profile yet? <Link to="/signup">Create one</Link></>}
            {' · '}<Link to="/app/live">Skip and just translate</Link>
          </p>
        </form>
      </div>
      <div className="auth-art">
        <div>
          <div style={{ fontSize: 64 }} aria-hidden="true">🤟</div>
          <h2>Technology for a more inclusive world</h2>
          <p className="note">
            Built on the INCLUDE dataset, signed by deaf signers from a school for the deaf.
          </p>
        </div>
      </div>
    </div>
  )
}

export const Login = () => <AuthPage mode="login" />
export const Signup = () => <AuthPage mode="signup" />
