import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { cancelRun, exportRunUrl, fetchRun } from '../api'
import type { SuiteRun, TestResult } from '../types'
import StatusBadge from './StatusBadge'

function passRate(run: SuiteRun): string {
  if (run.total_tests === 0) return '—'
  return `${((run.passed_tests / run.total_tests) * 100).toFixed(1)}%`
}

function duration(run: SuiteRun): string {
  if (!run.completed_at) return '—'
  const ms = new Date(run.completed_at).getTime() - new Date(run.started_at).getTime()
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`
}

function FailureSummary({ results }: { results: TestResult[] }) {
  const failing = results.filter(r => r.error !== null || !r.passed)
  if (failing.length === 0) return null

  // Aggregate criterion scores across failing tests to find patterns.
  const criterionTotals: Record<string, { sum: number; count: number }> = {}
  for (const r of failing) {
    for (const cs of r.criterion_scores) {
      if (!criterionTotals[cs.name]) criterionTotals[cs.name] = { sum: 0, count: 0 }
      criterionTotals[cs.name].sum += cs.score
      criterionTotals[cs.name].count += 1
    }
  }

  const worstCriteria = Object.entries(criterionTotals)
    .map(([name, { sum, count }]) => ({ name, avg: sum / count }))
    .sort((a, b) => a.avg - b.avg)
    .slice(0, 4)

  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
      <h2 className="text-sm font-semibold text-red-800 mb-3">
        {failing.length} failing test{failing.length !== 1 ? 's' : ''}
      </h2>
      {worstCriteria.length > 0 && (
        <>
          <p className="text-xs text-red-700 font-medium mb-2">Weakest criteria (avg across failures)</p>
          <div className="flex flex-wrap gap-2">
            {worstCriteria.map(c => (
              <span
                key={c.name}
                className={`text-xs px-2 py-1 rounded font-medium ${c.avg < 0.5 ? 'bg-red-200 text-red-900' : 'bg-orange-100 text-orange-800'}`}
              >
                {c.name}: {c.avg.toFixed(2)}
              </span>
            ))}
          </div>
        </>
      )}
      {failing.some(r => r.error !== null) && (
        <p className="text-xs text-yellow-700 mt-2">
          ⚠ {failing.filter(r => r.error !== null).length} test(s) errored — expand rows below for details.
        </p>
      )}
    </div>
  )
}

function ResultRow({ result }: { result: TestResult }) {
  const [expanded, setExpanded] = useState(false)
  const sortedCriteria = [...result.criterion_scores].sort((a, b) => a.score - b.score)

  return (
    <>
      <tr
        className={`hover:bg-gray-50 cursor-pointer transition-colors ${!result.passed && !result.error ? 'bg-red-50/40' : result.error ? 'bg-yellow-50/40' : ''}`}
        onClick={() => setExpanded(v => !v)}
      >
        <td className="px-4 py-3 font-mono text-xs text-gray-600">{result.test_id}</td>
        <td className="px-4 py-3">
          <StatusBadge passed={result.passed} error={result.error} />
        </td>
        <td className="px-4 py-3 text-right tabular-nums">
          {result.error ? (
            <span className="text-gray-400">—</span>
          ) : (
            <span className={result.passed ? 'text-green-700' : 'text-red-700'}>
              {result.weighted_score.toFixed(2)}
              {result.passing_threshold !== null && (
                <span className="text-gray-400 text-xs ml-1">/ {result.passing_threshold.toFixed(2)}</span>
              )}
            </span>
          )}
        </td>
        <td className="px-4 py-3 text-xs text-gray-500">
          {result.criterion_scores.map(c => `${c.name}: ${c.score.toFixed(2)}`).join('  ·  ')}
        </td>
        <td className="px-4 py-3 text-right text-xs text-gray-400">{expanded ? '▲' : '▼'}</td>
      </tr>
      {expanded && (
        <tr className="bg-gray-50">
          <td colSpan={5} className="px-4 pb-4 pt-2">
            <div className="space-y-3 text-sm">
              <div>
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Prompt</p>
                <p className="text-gray-700 whitespace-pre-wrap">{result.prompt}</p>
              </div>
              {result.error ? (
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Error</p>
                  <p className="text-yellow-700 bg-yellow-50 rounded px-3 py-2">{result.error}</p>
                </div>
              ) : (
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Model output</p>
                  <p className="text-gray-700 whitespace-pre-wrap bg-white border border-gray-200 rounded px-3 py-2">{result.raw_output}</p>
                </div>
              )}
              {sortedCriteria.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Criterion scores</p>
                  <div className="space-y-2">
                    {sortedCriteria.map(c => (
                      <div key={c.name} className="flex gap-3 items-start">
                        <span className={`font-medium w-28 shrink-0 text-xs pt-0.5 ${c.score < 0.6 ? 'text-red-700' : 'text-gray-700'}`}>
                          {c.name}
                        </span>
                        <span className={`tabular-nums w-10 shrink-0 font-medium ${c.score < 0.6 ? 'text-red-600' : 'text-gray-600'}`}>
                          {c.score.toFixed(2)}
                        </span>
                        <span className="text-gray-500 text-xs leading-relaxed">{c.reasoning}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const [run, setRun] = useState<SuiteRun | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [prevLoading, setPrevLoading] = useState(false)
  const [cancelling, setCancelling] = useState(false)

  useEffect(() => {
    if (!runId) return
    fetchRun(runId)
      .then(setRun)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [runId])

  async function handleCancel() {
    if (!runId || !run) return
    if (!confirm(`Cancel run ${runId.slice(0, 8)}?`)) return
    setCancelling(true)
    try {
      await cancelRun(runId)
      const updated = await fetchRun(runId)
      setRun(updated)
    } catch (e: unknown) {
      alert((e as Error).message)
    } finally {
      setCancelling(false)
    }
  }

  async function compareToPrevious() {
    if (!runId) return
    setPrevLoading(true)
    try {
      const prev = await fetch(`/api/runs/${runId}/previous`).then(r => {
        if (!r.ok) throw new Error('No previous run found for this suite.')
        return r.json()
      })
      navigate(`/diff?a=${prev.run_id}&b=${runId}`)
    } catch (e: unknown) {
      alert((e as Error).message)
    } finally {
      setPrevLoading(false)
    }
  }

  if (loading) return <p className="text-gray-500">Loading…</p>
  if (error) return <p className="text-red-600">Error: {error}</p>
  if (!run) return null

  const allPass = run.passed_tests === run.total_tests && run.errored_tests === 0

  return (
    <div>
      <Link to="/" className="text-sm text-indigo-600 hover:underline mb-4 inline-block">← All runs</Link>

      <div className="bg-white rounded-lg border border-gray-200 p-5 mb-6">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-xl font-semibold">{run.suite_name}</h1>
            <p className="text-sm text-gray-500 mt-0.5">v{run.suite_version} · {run.model}</p>
          </div>
          <div className="flex items-center gap-2">
            {(run.status === 'pending' || run.status === 'running') && (
              <button
                onClick={handleCancel}
                disabled={cancelling}
                className="text-sm px-3 py-1.5 rounded border border-orange-300 text-orange-700 hover:bg-orange-50 transition-colors disabled:opacity-40 font-medium"
              >
                {cancelling ? 'Cancelling…' : 'Cancel run'}
              </button>
            )}
            <a
              href={exportRunUrl(run.run_id, 'csv')}
              download
              className="text-sm px-3 py-1.5 rounded border border-gray-300 hover:bg-gray-50 transition-colors font-medium"
            >
              ↓ CSV
            </a>
            <a
              href={exportRunUrl(run.run_id, 'json')}
              download
              className="text-sm px-3 py-1.5 rounded border border-gray-300 hover:bg-gray-50 transition-colors font-medium"
            >
              ↓ JSON
            </a>
            <button
              onClick={compareToPrevious}
              disabled={prevLoading || run.status !== 'completed'}
              className="text-sm px-3 py-1.5 rounded border border-gray-300 hover:bg-gray-50 transition-colors disabled:opacity-40 font-medium"
            >
              {prevLoading ? 'Loading…' : 'Compare with previous'}
            </button>
            <span className={`text-2xl font-bold tabular-nums ${allPass ? 'text-green-600' : 'text-red-600'}`}>
              {passRate(run)}
            </span>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-4 gap-4 text-sm">
          {[
            ['Total', run.total_tests],
            ['Passed', run.passed_tests],
            ['Failed', run.failed_tests],
            ['Errors', run.errored_tests],
          ].map(([label, val]) => (
            <div key={label as string} className="bg-gray-50 rounded p-3">
              <p className="text-xs text-gray-500 mb-1">{label}</p>
              <p className="text-lg font-semibold">{val}</p>
            </div>
          ))}
        </div>
        <div className="mt-3 flex gap-6 text-xs text-gray-500 flex-wrap">
          <span>Judge: {run.judge_model}</span>
          <span>Duration: {duration(run)}</span>
          {run.suite_path && <span>Suite: {run.suite_path}</span>}
          {run.tags.length > 0 && <span>Tags: {run.tags.join(', ')}</span>}
          <span className="font-mono">{run.run_id}</span>
        </div>
        {Object.keys(run.labels ?? {}).length > 0 && (
          <div className="mt-2 flex gap-1.5 flex-wrap">
            {Object.entries(run.labels).map(([k, v]) => (
              <span key={k} className="text-xs bg-indigo-50 text-indigo-600 px-2 py-0.5 rounded font-mono">
                {k}={v}
              </span>
            ))}
          </div>
        )}
        {run.error_message && (
          <p className="mt-3 text-sm text-red-700 bg-red-50 rounded px-3 py-2">{run.error_message}</p>
        )}
      </div>

      {run.status === 'completed' && <FailureSummary results={run.results} />}

      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-4 py-3 text-left font-medium text-gray-500">Test ID</th>
              <th className="px-4 py-3 text-left font-medium text-gray-500">Status</th>
              <th className="px-4 py-3 text-right font-medium text-gray-500">Score</th>
              <th className="px-4 py-3 text-left font-medium text-gray-500">Criteria</th>
              <th className="px-4 py-3 w-8"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {run.results.map(r => <ResultRow key={r.test_id} result={r} />)}
          </tbody>
        </table>
      </div>
    </div>
  )
}
