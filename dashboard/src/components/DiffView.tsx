import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { fetchDiff, fetchRun } from '../api'
import type { RunSummary, TestDiff } from '../types'
import StatusBadge from './StatusBadge'

function ChangeIndicator({ diff }: { diff: TestDiff }) {
  if (diff.is_regression) return <span className="text-red-600 font-bold" title="Regression">▼</span>
  if (diff.is_improvement) return <span className="text-green-600 font-bold" title="Improvement">▲</span>
  if (diff.result_a === null) return <span className="text-blue-500 font-bold" title="New test">+</span>
  if (diff.result_b === null) return <span className="text-gray-400 font-bold" title="Removed test">−</span>
  return <span className="text-gray-300">→</span>
}

function DeltaCell({ delta }: { delta: number | null }) {
  if (delta === null) return <span className="text-gray-400">—</span>
  const sign = delta > 0 ? '+' : ''
  const color = delta > 0.02 ? 'text-green-600' : delta < -0.02 ? 'text-red-600' : 'text-gray-500'
  return <span className={`tabular-nums font-medium ${color}`}>{sign}{delta.toFixed(2)}</span>
}

function TopRegressions({ diffs }: { diffs: TestDiff[] }) {
  const regressions = diffs
    .filter(d => d.is_regression && d.score_delta !== null)
    .sort((a, b) => (a.score_delta ?? 0) - (b.score_delta ?? 0))
    .slice(0, 5)

  if (regressions.length === 0) return null

  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
      <h2 className="text-sm font-semibold text-red-800 mb-3">Top regressions</h2>
      <div className="space-y-2">
        {regressions.map(d => {
          // Find criteria that dropped most between A and B.
          const deltasByCriterion: { name: string; delta: number }[] = []
          if (d.result_a && d.result_b) {
            const scoresA = Object.fromEntries(d.result_a.criterion_scores.map(c => [c.name, c.score]))
            for (const cs of d.result_b.criterion_scores) {
              const prev = scoresA[cs.name]
              if (prev !== undefined) deltasByCriterion.push({ name: cs.name, delta: cs.score - prev })
            }
            deltasByCriterion.sort((a, b) => a.delta - b.delta)
          }

          return (
            <div key={d.test_id} className="flex items-start gap-3">
              <span className="font-mono text-xs text-red-700 w-48 shrink-0 pt-0.5">{d.test_id}</span>
              <span className="text-red-600 font-medium tabular-nums text-xs pt-0.5 w-12 shrink-0">
                {d.score_delta !== null && d.score_delta < 0 ? d.score_delta.toFixed(2) : ''}
              </span>
              <div className="flex flex-wrap gap-1">
                {deltasByCriterion.slice(0, 3).map(c => (
                  <span
                    key={c.name}
                    className={`text-xs px-1.5 py-0.5 rounded ${c.delta < -0.1 ? 'bg-red-200 text-red-900' : 'bg-orange-100 text-orange-800'}`}
                  >
                    {c.name}: {c.delta > 0 ? '+' : ''}{c.delta.toFixed(2)}
                  </span>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function DiffView() {
  const [searchParams] = useSearchParams()
  const runIdA = searchParams.get('a') ?? ''
  const runIdB = searchParams.get('b') ?? ''

  const [diffs, setDiffs] = useState<TestDiff[]>([])
  const [runA, setRunA] = useState<RunSummary | null>(null)
  const [runB, setRunB] = useState<RunSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!runIdA || !runIdB) {
      setLoading(false)
      return
    }
    Promise.all([fetchDiff(runIdA, runIdB), fetchRun(runIdA), fetchRun(runIdB)])
      .then(([d, a, b]) => { setDiffs(d); setRunA(a); setRunB(b) })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [runIdA, runIdB])

  if (!runIdA || !runIdB) {
    return (
      <div className="text-center py-20 text-gray-400">
        <p>Select two runs from the <Link to="/" className="text-indigo-600 hover:underline">run list</Link> to compare.</p>
      </div>
    )
  }

  if (loading) return <p className="text-gray-500">Loading diff…</p>
  if (error) return <p className="text-red-600">Error: {error}</p>

  const regressions = diffs.filter(d => d.is_regression).length
  const improvements = diffs.filter(d => d.is_improvement).length
  const unchanged = diffs.filter(d => !d.is_regression && !d.is_improvement && d.result_a && d.result_b).length

  // Sort: regressions first (worst delta first), then improvements, then unchanged.
  const sorted = [...diffs].sort((a, b) => {
    if (a.is_regression && !b.is_regression) return -1
    if (!a.is_regression && b.is_regression) return 1
    if (a.is_regression && b.is_regression) return (a.score_delta ?? 0) - (b.score_delta ?? 0)
    if (a.is_improvement && !b.is_improvement) return -1
    if (!a.is_improvement && b.is_improvement) return 1
    return 0
  })

  return (
    <div>
      <Link to="/" className="text-sm text-indigo-600 hover:underline mb-4 inline-block">← All runs</Link>

      <div className="grid grid-cols-2 gap-4 mb-6">
        {[['A — Baseline', runA], ['B — Candidate', runB]].map(([label, run]) => (
          <div key={label as string} className="bg-white rounded-lg border border-gray-200 p-4">
            <p className="text-xs text-gray-500 font-medium uppercase tracking-wide mb-1">{label as string}</p>
            <p className="font-semibold">{(run as RunSummary)?.suite_name ?? '—'}</p>
            <p className="text-sm text-gray-500">{(run as RunSummary)?.model}</p>
            <p className="font-mono text-xs text-gray-400 mt-1">{(run as RunSummary)?.run_id.slice(0, 8)}…</p>
          </div>
        ))}
      </div>

      <div className="flex gap-4 mb-6 text-sm">
        {regressions > 0 && (
          <span className="bg-red-50 text-red-700 rounded px-3 py-1.5 font-medium">
            ▼ {regressions} regression{regressions !== 1 ? 's' : ''}
          </span>
        )}
        {improvements > 0 && (
          <span className="bg-green-50 text-green-700 rounded px-3 py-1.5 font-medium">
            ▲ {improvements} improvement{improvements !== 1 ? 's' : ''}
          </span>
        )}
        <span className="bg-gray-100 text-gray-600 rounded px-3 py-1.5">
          → {unchanged} unchanged
        </span>
      </div>

      <TopRegressions diffs={diffs} />

      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-4 py-3 text-left font-medium text-gray-500 w-6"></th>
              <th className="px-4 py-3 text-left font-medium text-gray-500">Test ID</th>
              <th className="px-4 py-3 text-center font-medium text-gray-500">A status</th>
              <th className="px-4 py-3 text-right font-medium text-gray-500">A score</th>
              <th className="px-4 py-3 text-center font-medium text-gray-500">B status</th>
              <th className="px-4 py-3 text-right font-medium text-gray-500">B score</th>
              <th className="px-4 py-3 text-right font-medium text-gray-500">Δ</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sorted.map(diff => (
              <tr
                key={diff.test_id}
                className={
                  diff.is_regression ? 'bg-red-50' :
                  diff.is_improvement ? 'bg-green-50' : ''
                }
              >
                <td className="px-4 py-3 text-center">
                  <ChangeIndicator diff={diff} />
                </td>
                <td className="px-4 py-3 font-mono text-xs text-gray-600">{diff.test_id}</td>
                <td className="px-4 py-3 text-center">
                  {diff.result_a
                    ? <StatusBadge passed={diff.result_a.passed} error={diff.result_a.error} />
                    : <span className="text-gray-300 text-xs">—</span>}
                </td>
                <td className="px-4 py-3 text-right tabular-nums text-gray-600">
                  {diff.result_a && !diff.result_a.error ? diff.result_a.weighted_score.toFixed(2) : '—'}
                </td>
                <td className="px-4 py-3 text-center">
                  {diff.result_b
                    ? <StatusBadge passed={diff.result_b.passed} error={diff.result_b.error} />
                    : <span className="text-gray-300 text-xs">—</span>}
                </td>
                <td className="px-4 py-3 text-right tabular-nums text-gray-600">
                  {diff.result_b && !diff.result_b.error ? diff.result_b.weighted_score.toFixed(2) : '—'}
                </td>
                <td className="px-4 py-3 text-right">
                  <DeltaCell delta={diff.score_delta} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
