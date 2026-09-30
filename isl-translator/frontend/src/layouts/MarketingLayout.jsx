import { Link, NavLink, Outlet } from 'react-router-dom'
import { useApp } from '../store.jsx'

/** Public pages: landing, login, sign up. */
export default function MarketingLayout() {
  const { profile } = useApp()
  return (
    <>
      <header className="appbar">
        <Link to="/" className="brand" style={{ textDecoration: 'none', color: 'inherit' }}>
          <div className="brand-mark" aria-hidden="true">SB</div>
          <div>
            <h1>SignBridge AI</h1>
            <p>Indian Sign Language → English, in real time</p>
          </div>
        </Link>
        <nav>
          <NavLink to="/" end className={({ isActive }) => (isActive ? 'active' : '')}>Home</NavLink>
          <NavLink to="/app/live" className={({ isActive }) => (isActive ? 'active' : '')}>Live translation</NavLink>
          <NavLink to="/app/healthcare" className={({ isActive }) => (isActive ? 'active' : '')}>Healthcare</NavLink>
          <NavLink to="/app/learn" className={({ isActive }) => (isActive ? 'active' : '')}>Learn signs</NavLink>
          {profile
            ? <Link to="/app/dashboard" className="active">Open app</Link>
            : <><Link to="/login">Login</Link><Link to="/signup" className="active">Get started</Link></>}
        </nav>
      </header>
      <Outlet />
      <footer>
        Assistive demo, not a medical device. Reference signs come from the INCLUDE dataset
        (AI4Bharat, CC BY 4.0), performed by deaf signers.
      </footer>
    </>
  )
}
