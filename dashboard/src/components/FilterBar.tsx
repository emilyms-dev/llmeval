import type { RunFilters } from '../api'

const STATUSES = ['', 'completed', 'failed', 'running', 'pending'] as const

interface Props {
  filters: RunFilters
  suites: string[]
  onChange: (f: RunFilters) => void
}

export default function FilterBar({ filters, suites, onChange }: Props) {
  function set(patch: Partial<RunFilters>) {
    onChange({ ...filters, ...patch, offset: 0 })
  }

  const activeCount = [
    filters.suite,
    filters.model,
    filters.status,
    filters.tag,
    filters.date_from,
    filters.date_to,
  ].filter(Boolean).length

  return (
    <div className="mb-5">
      <div className="flex flex-wrap gap-2 items-end">
        {/* Suite */}
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500 font-medium">Suite</label>
          <select
            value={filters.suite ?? ''}
            onChange={e => set({ suite: e.target.value || undefined })}
            className="text-sm border border-gray-300 rounded px-2 py-1.5 bg-white min-w-[160px]"
          >
            <option value="">All suites</option>
            {suites.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        {/* Model */}
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500 font-medium">Model</label>
          <input
            type="text"
            placeholder="e.g. claude-sonnet…"
            value={filters.model ?? ''}
            onChange={e => set({ model: e.target.value || undefined })}
            className="text-sm border border-gray-300 rounded px-2 py-1.5 min-w-[180px]"
          />
        </div>

        {/* Status */}
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500 font-medium">Status</label>
          <select
            value={filters.status ?? ''}
            onChange={e => set({ status: e.target.value || undefined })}
            className="text-sm border border-gray-300 rounded px-2 py-1.5 bg-white"
          >
            {STATUSES.map(s => (
              <option key={s} value={s}>{s || 'All statuses'}</option>
            ))}
          </select>
        </div>

        {/* Tag */}
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500 font-medium">Tag</label>
          <input
            type="text"
            placeholder="tag substring…"
            value={filters.tag ?? ''}
            onChange={e => set({ tag: e.target.value || undefined })}
            className="text-sm border border-gray-300 rounded px-2 py-1.5 min-w-[140px]"
          />
        </div>

        {/* Date from */}
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500 font-medium">From</label>
          <input
            type="date"
            value={filters.date_from ?? ''}
            onChange={e => set({ date_from: e.target.value || undefined })}
            className="text-sm border border-gray-300 rounded px-2 py-1.5"
          />
        </div>

        {/* Date to */}
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500 font-medium">To</label>
          <input
            type="date"
            value={filters.date_to ?? ''}
            onChange={e => set({ date_to: e.target.value || undefined })}
            className="text-sm border border-gray-300 rounded px-2 py-1.5"
          />
        </div>

        {/* Clear */}
        {activeCount > 0 && (
          <button
            onClick={() =>
              onChange({ limit: filters.limit, offset: 0 })
            }
            className="text-sm text-indigo-600 hover:underline self-end pb-1.5"
          >
            Clear {activeCount} filter{activeCount !== 1 ? 's' : ''}
          </button>
        )}
      </div>
    </div>
  )
}
