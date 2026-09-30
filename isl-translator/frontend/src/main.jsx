import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppProvider } from './store.jsx'
import MarketingLayout from './layouts/MarketingLayout.jsx'
import AppLayout from './layouts/AppLayout.jsx'
import Landing from './pages/Landing.jsx'
import { Login, Signup } from './pages/Auth.jsx'
import Dashboard from './pages/Dashboard.jsx'
import LiveTranslation from './pages/LiveTranslation.jsx'
import Healthcare from './pages/Healthcare.jsx'
import Learn from './pages/Learn.jsx'
import History from './pages/History.jsx'
import Teach from './pages/Teach.jsx'
import Fingerspell from './pages/Fingerspell.jsx'
import { EmergencyInterpreter, FindInterpreter, InterpreterConnect } from './pages/InterpreterConnect.jsx'
import InterpreterCall from './pages/InterpreterCall.jsx'
import InterpreterDashboard from './pages/InterpreterDashboard.jsx'
import Settings from './pages/Settings.jsx'
import './styles.css'
import './shell.css'
import './mobile.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<MarketingLayout />}>
            <Route path="/" element={<Landing />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
          </Route>

          <Route path="/app" element={<AppLayout />}>
            <Route index element={<Navigate to="/app/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="live" element={<LiveTranslation />} />
            <Route path="healthcare" element={<Healthcare />} />
            <Route
              path="conversation"
              element={
                <LiveTranslation
                  mode="general"
                  title="General conversation"
                  lead="Everyday two-way conversation using ISL and speech."
                />
              }
            />
            <Route path="learn" element={<Learn />} />
            <Route path="teach" element={<Teach />} />
            <Route path="fingerspell" element={<Fingerspell />} />
            <Route path="interpreter" element={<InterpreterConnect />} />
            <Route path="interpreter/find" element={<FindInterpreter />} />
            <Route path="interpreter/emergency" element={<EmergencyInterpreter />} />
            <Route path="interpreter/call/:roomId" element={<InterpreterCall />} />
            <Route path="interpreter/dashboard" element={<InterpreterDashboard />} />
            <Route path="history" element={<History />} />
            <Route path="settings" element={<Settings />} />
          </Route>

          {/* old links keep working */}
          <Route path="/learn" element={<Navigate to="/app/learn" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AppProvider>
  </React.StrictMode>,
)
