// Shared formatting helpers for table cells.

export function fmtCost(usd: number | null | undefined): string {
  if (usd == null) return '—'
  if (usd === 0) return '$0'
  if (usd < 0.01) return `$${usd.toFixed(4)}`
  return `$${usd.toFixed(2)}`
}

export function fmtScore(v: number | undefined): string {
  if (v == null) return '—'
  return v.toFixed(4)
}
