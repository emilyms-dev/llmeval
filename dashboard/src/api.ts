import type { RunSummary, SuiteRun, TestDiff } from './types'

const BASE = '/api'

function authHeaders(): Record<string, string> {
  const token = import.meta.env.VITE_LLMEVAL_API_TOKEN
  return token ? { Authorization: `Bearer ${token}` } : {}
}

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
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
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

export interface RunFilters {
  suite?: string
  model?: string
  status?: string
  tag?: string
  tag_match?: 'exact' | 'fuzzy'
  date_from?: string
  date_to?: string
  limit?: number
  offset?: number
}

export function fetchSuites(): Promise<string[]> {
  return get<string[]>('/suites')
}

export function fetchRuns(filters?: RunFilters): Promise<RunSummary[]> {
  const q = new URLSearchParams()
  if (filters?.suite) q.set('suite', filters.suite)
  if (filters?.model) q.set('model', filters.model)
  if (filters?.status) q.set('status', filters.status)
  if (filters?.tag) q.set('tag', filters.tag)
  if (filters?.tag_match) q.set('tag_match', filters.tag_match)
  if (filters?.date_from) q.set('date_from', filters.date_from)
  if (filters?.date_to) q.set('date_to', filters.date_to)
  if (filters?.limit != null) q.set('limit', String(filters.limit))
  if (filters?.offset != null) q.set('offset', String(filters.offset))
  const qs = q.toString()
  return get<RunSummary[]>(`/runs${qs ? `?${qs}` : ''}`)
}

export function fetchRun(runId: string): Promise<SuiteRun> {
  return get<SuiteRun>(`/runs/${runId}`)
}

export function fetchPreviousRun(runId: string): Promise<SuiteRun> {
  return get<SuiteRun>(`/runs/${runId}/previous`)
}

export function triggerRun(body: {
  suite_path: string
  model?: string
  tags?: string[]
  concurrency?: number
}): Promise<{ run_id: string; status: string }> {
  return post('/runs', body)
}

export function cancelRun(runId: string): Promise<{ run_id: string; status: string }> {
  return post(`/runs/${runId}/cancel`, {})
}

export function fetchDiff(runIdA: string, runIdB: string): Promise<TestDiff[]> {
  return get<TestDiff[]>(`/runs/${runIdA}/diff/${runIdB}`)
}

export function exportRunUrl(runId: string, format: 'json' | 'csv'): string {
  return `${BASE}/runs/${runId}/export?format=${format}`
}
