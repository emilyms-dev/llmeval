interface Props {
  passed: boolean
  error: string | null
  size?: 'sm' | 'md'
}

export default function StatusBadge({ passed, error, size = 'sm' }: Props) {
  const base = size === 'sm'
    ? 'inline-block rounded px-1.5 py-0.5 text-xs font-medium'
    : 'inline-block rounded px-2 py-1 text-sm font-semibold'

  if (error) {
    return <span className={`${base} bg-yellow-100 text-yellow-800`}>Error</span>
  }
  if (passed) {
    return <span className={`${base} bg-green-100 text-green-800`}>Pass</span>
  }
  return <span className={`${base} bg-red-100 text-red-800`}>Fail</span>
}
