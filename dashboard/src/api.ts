import type { RunSummary, SuiteRun, TestDiff } from './types'

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const { detail } = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail ?? res.statusText)
  }
  return res.json() as Promise<T>
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const json = await res.json().catch(() => ({ detail: res.statusText }))
    const detail = Array.isArray(json.detail)
      ? json.detail.map((e: { msg: string }) => e.msg).join(', ')
      : (json.detail ?? res.statusText)
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export function fetchSuites(): Promise<string[]> {
  return get<string[]>('/suites')
}

export function fetchRuns(params?: {
  suite?: string
  limit?: number
  offset?: number
}): Promise<RunSummary[]> {
  const q = new URLSearchParams()
  if (params?.suite) q.set('suite', params.suite)
  if (params?.limit != null) q.set('limit', String(params.limit))
  if (params?.offset != null) q.set('offset', String(params.offset))
  const qs = q.toString()
  return get<RunSummary[]>(`/runs${qs ? `?${qs}` : ''}`)
}

export function fetchRun(runId: string): Promise<SuiteRun> {
  return get<SuiteRun>(`/runs/${runId}`)
}

export function triggerRun(body: {
  suite_path: string
  model?: string
  tags?: string[]
  concurrency?: number
}): Promise<{ run_id: string; status: string }> {
  return post('/runs', body)
}

export function fetchPreviousRun(runId: string): Promise<SuiteRun> {
  return get<SuiteRun>(`/runs/${runId}/previous`)
}

export function fetchDiff(runIdA: string, runIdB: string): Promise<TestDiff[]> {
  return get<TestDiff[]>(`/runs/${runIdA}/diff/${runIdB}`)
}
