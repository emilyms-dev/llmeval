import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import RunList from './components/RunList'
import RunDetail from './components/RunDetail'
import DiffView from './components/DiffView'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 text-gray-900">
        <header className="bg-white border-b border-gray-200 px-6 py-4">
          <a href="/" className="text-lg font-semibold tracking-tight hover:text-indigo-600 transition-colors">
            llmeval
          </a>
        </header>
        <main className="max-w-6xl mx-auto px-6 py-8">
          <Routes>
            <Route path="/" element={<RunList />} />
            <Route path="/runs/:runId" element={<RunDetail />} />
            <Route path="/diff" element={<DiffView />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
