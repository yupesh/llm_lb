import { useMemo, useState } from 'react'
import type { Index, MatrixEntry } from '../types'

// Judge/reference models — scored indirectly as LLM-as-Judge, not system-under-test.
// Keep them in index.json for attribution but hide from the compare leaderboard.
const HIDDEN_MODEL_IDS = new Set(['dummy@local', 'judge@openai', 'gpt-4o-mini@openai'])

function minMaxNormalize(values: number[]): number[] {
  if (!values.length) return []
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min
  if (range === 0) return values.map(() => 1)
  return values.map((v) => (v - min) / range)
}

export function CompareView({ index }: { index: Index }) {
  const visibleModels = useMemo(
    () => index.models.filter((m) => !HIDDEN_MODEL_IDS.has(m.id)),
    [index.models],
  )

  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(
    () => new Set(index.tasks.map((t) => t.id)),
  )
  const [selectedModels, setSelectedModels] = useState<Set<string>>(
    () => new Set(visibleModels.map((m) => m.id)),
  )

  const tasks = index.tasks.filter((t) => selectedTasks.has(t.id))
  const models = visibleModels.filter((m) => selectedModels.has(m.id))

  // model_id -> task_id -> entry
  const lookup = useMemo(() => {
    const m: Record<string, Record<string, MatrixEntry>> = {}
    for (const e of index.matrix) {
      ;(m[e.model_id] ??= {})[e.task_id] = e
    }
    return m
  }, [index])

  // Per task: normalize scores across selected models, and record each
  // model's rank within the column. Per model: average of its normalized
  // scores across selected tasks. This makes scores from different metric
  // scales (accuracy, F1, ...) comparable in one column.
  const { normalizedAvg, perTask } = useMemo(() => {
    const collected: Record<string, number[]> = {}
    const perTask: Record<string, { norm: Record<string, number>; rank: Record<string, number> }> = {}
    for (const t of tasks) {
      const present = models
        .map((m) => ({ id: m.id, score: lookup[m.id]?.[t.id]?.score }))
        .filter((x): x is { id: string; score: number } => typeof x.score === 'number')
      if (!present.length) continue
      const norm = minMaxNormalize(present.map((p) => p.score))
      const normMap: Record<string, number> = {}
      present.forEach((p, i) => {
        normMap[p.id] = norm[i]
        ;(collected[p.id] ??= []).push(norm[i])
      })
      // Tie-aware ranking: equal scores share a rank (1, 1, 3, 4, ...).
      // Lets us bold every leader and underline every runner-up — the
      // convention in benchmark papers where "best" can be a tie.
      const rankMap: Record<string, number> = {}
      const sorted = [...present].sort((a, b) => b.score - a.score)
      let prev: number | null = null
      let prevRank = 0
      sorted.forEach((p, i) => {
        const rank = prev !== null && p.score === prev ? prevRank : i + 1
        rankMap[p.id] = rank
        prev = p.score
        prevRank = rank
      })
      perTask[t.id] = { norm: normMap, rank: rankMap }
    }
    const avg: Record<string, number> = {}
    for (const [k, arr] of Object.entries(collected)) {
      avg[k] = arr.reduce((a, b) => a + b, 0) / arr.length
    }
    return { normalizedAvg: avg, perTask }
  }, [tasks, models, lookup])

  // Row-level normalization of the Avg column for the leader tint.
  const avgNorm = useMemo(() => {
    const vals = models.map((m) => normalizedAvg[m.id]).filter((v) => typeof v === 'number') as number[]
    if (!vals.length) return {} as Record<string, number>
    const min = Math.min(...vals)
    const max = Math.max(...vals)
    const range = max - min
    const out: Record<string, number> = {}
    for (const m of models) {
      const v = normalizedAvg[m.id]
      if (typeof v === 'number') out[m.id] = range === 0 ? 1 : (v - min) / range
    }
    return out
  }, [models, normalizedAvg])

  const sortedModels = [...models].sort(
    (a, b) => (normalizedAvg[b.id] ?? -1) - (normalizedAvg[a.id] ?? -1),
  )

  function toggle(set: Set<string>, value: string, setter: (s: Set<string>) => void) {
    const next = new Set(set)
    if (next.has(value)) next.delete(value)
    else next.add(value)
    setter(next)
  }

  return (
    <div className="compare">
      <details open>
        <summary>Filters</summary>
        <div className="filters">
          <fieldset>
            <legend>Tasks ({selectedTasks.size}/{index.tasks.length})</legend>
            {index.tasks.map((t) => (
              <label key={t.id}>
                <input
                  type="checkbox"
                  checked={selectedTasks.has(t.id)}
                  onChange={() => toggle(selectedTasks, t.id, setSelectedTasks)}
                />{' '}
                {t.id}
              </label>
            ))}
          </fieldset>
          <fieldset>
            <legend>Models ({selectedModels.size}/{visibleModels.length})</legend>
            {visibleModels.map((m) => (
              <label key={m.id}>
                <input
                  type="checkbox"
                  checked={selectedModels.has(m.id)}
                  onChange={() => toggle(selectedModels, m.id, setSelectedModels)}
                />{' '}
                {m.display_name}
              </label>
            ))}
          </fieldset>
        </div>
      </details>

      <div className="matrix-wrap">
        <table className="matrix">
          <thead>
            <tr>
              <th>Model</th>
              <th>Params</th>
              {tasks.map((t) => (
                <th key={t.id}>
                  {t.id}
                  <div className="metric">{t.primary_metric}</div>
                </th>
              ))}
              <th>Avg (norm)</th>
            </tr>
          </thead>
          <tbody>
            {sortedModels.map((m) => (
              <tr key={m.id}>
                <th scope="row" className="model-cell" title={m.hf_uri ?? undefined}>
                  <code>{m.id}</code>
                  {m.hf_uri && <div className="hf-uri">{m.hf_uri}</div>}
                </th>
                <td className="params-cell">{m.params ?? '—'}</td>
                {tasks.map((t) => {
                  const e = lookup[m.id]?.[t.id]
                  const info = perTask[t.id]
                  const n = info?.norm[m.id]
                  const rank = info?.rank[m.id]
                  const style =
                    typeof n === 'number'
                      ? { backgroundColor: `rgba(34, 197, 94, ${(n * 0.32).toFixed(3)})` }
                      : undefined
                  const className = [
                    e ? '' : 'muted',
                    rank === 1 ? 'leader' : '',
                    rank === 2 ? 'runner-up' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')
                  return (
                    <td key={t.id} className={className} style={style}>
                      {e ? e.score.toFixed(3) : '—'}
                    </td>
                  )
                })}
                {(() => {
                  const v = normalizedAvg[m.id]
                  const n = avgNorm[m.id]
                  const style =
                    typeof n === 'number'
                      ? { backgroundColor: `rgba(37, 99, 235, ${(n * 0.28).toFixed(3)})` }
                      : undefined
                  return (
                    <td className="avg-cell" style={style}>
                      <strong>{(v ?? 0).toFixed(3)}</strong>
                    </td>
                  )
                })()}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
