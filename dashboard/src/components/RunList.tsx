import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { fetchRuns } from '../api'
import type { RunSummary } from '../types'
import NewRunModal from './NewRunModal'

function passRate(run: RunSummary): string {
  if (run.status !== 'completed') return '—'
  if (run.total_tests === 0) return '0 tests'
  return `${((run.passed_tests / run.total_tests) * 100).toFixed(1)}%`
}

function StatusPill({ status }: { status: string }) {
  const styles: Record<string, string> = {
    pending: 'bg-gray-100 text-gray-500',
    running: 'bg-blue-50 text-blue-600',
    completed: 'bg-green-50 text-green-700',
    failed: 'bg-red-50 text-red-700',
  }
  const cls = styles[status] ?? 'bg-gray-100 text-gray-500'
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full ${cls}`}>
      {(status === 'pending' || status === 'running') && <Spinner />}
      {status}
    </span>
  )
}

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4 text-indigo-500 inline" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  )
}

export default function RunList() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string[]>([])
  const [showModal, setShowModal] = useState(false)
  const navigate = useNavigate()
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  function load() {
    return fetchRuns({ limit: 50 })
      .then(setRuns)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }

  // Start/stop polling based on whether any run is still in progress.
  useEffect(() => {
    load()
  }, [])

  useEffect(() => {
    const hasInProgress = runs.some(r => r.status === 'pending' || r.status === 'running')

    if (hasInProgress && !pollRef.current) {
      pollRef.current = setInterval(() => {
        fetchRuns({ limit: 50 }).then(setRuns).catch(() => {})
      }, 3000)
    } else if (!hasInProgress && pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [runs])

  function toggleSelect(runId: string) {
    setSelected(prev =>
      prev.includes(runId) ? prev.filter(id => id !== runId) : [...prev.slice(-1), runId]
    )
  }

  function handleRunStarted(runId: string) {
    // Immediately refresh and kick off polling.
    fetchRuns({ limit: 50 }).then(setRuns).catch(() => {})
    // Scroll hint — remove the run from selection if it was there.
    setSelected(prev => prev.filter(id => id !== runId))
  }

  if (loading) return <p className="text-gray-500">Loading runs…</p>
  if (error) return <p className="text-red-600">Error: {error}</p>

  return (
    <div>
      {showModal && (
        <NewRunModal
          onClose={() => setShowModal(false)}
          onStarted={handleRunStarted}
        />
      )}

      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold">Runs</h1>
        <div className="flex gap-2">
          <button
            onClick={() => navigate(`/diff?a=${selected[0]}&b=${selected[1]}`)}
            disabled={selected.length !== 2}
            className="px-3 py-1.5 text-sm rounded border border-gray-300 font-medium disabled:opacity-40 hover:bg-gray-50 transition-colors"
          >
            Compare selected
          </button>
          <button
            onClick={() => setShowModal(true)}
            className="px-3 py-1.5 text-sm rounded bg-indigo-600 text-white font-medium hover:bg-indigo-700 transition-colors"
          >
            + New run
          </button>
        </div>
      </div>

      {runs.length === 0 ? (
        <div className="text-center py-20 text-gray-400">
          <p className="text-lg font-medium">No runs yet.</p>
          <p className="text-sm mt-1">
            Click <strong>+ New run</strong> above or run{' '}
            <code className="bg-gray-100 px-1 rounded">llmeval run --suite your-suite.yaml</code>{' '}
            in the terminal.
          </p>
        </div>
      ) : (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left font-medium text-gray-500 w-8"></th>
                <th className="px-4 py-3 text-left font-medium text-gray-500">Run ID</th>
                <th className="px-4 py-3 text-left font-medium text-gray-500">Suite</th>
                <th className="px-4 py-3 text-left font-medium text-gray-500">Model</th>
                <th className="px-4 py-3 text-left font-medium text-gray-500">Status</th>
                <th className="px-4 py-3 text-right font-medium text-gray-500">Pass rate</th>
                <th className="px-4 py-3 text-right font-medium text-gray-500">Tests</th>
                <th className="px-4 py-3 text-right font-medium text-gray-500">Started</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {runs.map(run => {
                const inProgress = run.status === 'pending' || run.status === 'running'
                return (
                  <tr
                    key={run.run_id}
                    className={`transition-colors ${selected.includes(run.run_id) ? 'bg-indigo-50' : 'hover:bg-gray-50'}`}
                  >
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={selected.includes(run.run_id)}
                        onChange={() => toggleSelect(run.run_id)}
                        disabled={inProgress}
                        className="rounded border-gray-300 text-indigo-600 disabled:opacity-40"
                      />
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-gray-500">
                      {inProgress ? (
                        <span>{run.run_id.slice(0, 8)}…</span>
                      ) : (
                        <Link to={`/runs/${run.run_id}`} className="hover:text-indigo-600 hover:underline">
                          {run.run_id.slice(0, 8)}…
                        </Link>
                      )}
                    </td>
                    <td className="px-4 py-3 font-medium text-gray-700">{run.suite_name}</td>
                    <td className="px-4 py-3 text-gray-600">{run.model}</td>
                    <td className="px-4 py-3"><StatusPill status={run.status} /></td>
                    <td className="px-4 py-3 text-right">
                      {run.status === 'completed' ? (
                        <span className={`font-medium ${
                          run.errored_tests > 0 ? 'text-yellow-600'
                          : run.failed_tests > 0 ? 'text-red-600'
                          : run.total_tests > 0 ? 'text-green-600'
                          : 'text-gray-400'
                        }`}>
                          {passRate(run)}
                        </span>
                      ) : run.status === 'failed' ? (
                        <span className="text-red-500 text-xs">{run.error_message ?? 'failed'}</span>
                      ) : (
                        <span className="text-gray-400 italic text-xs">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right text-gray-600">
                      {run.status === 'completed' ? run.total_tests : '—'}
                    </td>
                    <td className="px-4 py-3 text-right text-gray-500 text-xs">
                      {fmtDate(run.started_at)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {selected.length > 0 && (
        <p className="mt-2 text-xs text-gray-500">
          {selected.length === 1
            ? 'Select one more run to compare.'
            : 'Two runs selected — click "Compare selected".'}
        </p>
      )}
    </div>
  )
}
