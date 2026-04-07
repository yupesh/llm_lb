import { useMemo, useState } from 'react'
import type { Index, MatrixEntry } from '../types'

function minMaxNormalize(values: number[]): number[] {
  if (!values.length) return []
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min
  if (range === 0) return values.map(() => 1)
  return values.map((v) => (v - min) / range)
}

export function CompareView({ index }: { index: Index }) {
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(
    () => new Set(index.tasks.map((t) => t.id)),
  )
  const [selectedModels, setSelectedModels] = useState<Set<string>>(
    () => new Set(index.models.map((m) => m.id)),
  )

  const tasks = index.tasks.filter((t) => selectedTasks.has(t.id))
  const models = index.models.filter((m) => selectedModels.has(m.id))

  // model_id -> task_id -> entry
  const lookup = useMemo(() => {
    const m: Record<string, Record<string, MatrixEntry>> = {}
    for (const e of index.matrix) {
      ;(m[e.model_id] ??= {})[e.task_id] = e
    }
    return m
  }, [index])

  // Per task: normalize scores across selected models. Per model: average of
  // its normalized scores across selected tasks. This makes scores from
  // different metric scales (accuracy, F1, ...) comparable in one column.
  const normalizedAvg = useMemo(() => {
    const collected: Record<string, number[]> = {}
    for (const t of tasks) {
      const present = models
        .map((m) => ({ id: m.id, score: lookup[m.id]?.[t.id]?.score }))
        .filter((x): x is { id: string; score: number } => typeof x.score === 'number')
      if (!present.length) continue
      const norm = minMaxNormalize(present.map((p) => p.score))
      present.forEach((p, i) => {
        ;(collected[p.id] ??= []).push(norm[i])
      })
    }
    const out: Record<string, number> = {}
    for (const [k, arr] of Object.entries(collected)) {
      out[k] = arr.reduce((a, b) => a + b, 0) / arr.length
    }
    return out
  }, [tasks, models, lookup])

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
            <legend>Models ({selectedModels.size}/{index.models.length})</legend>
            {index.models.map((m) => (
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
                <td>
                  <code>{m.id}</code>
                </td>
                {tasks.map((t) => {
                  const e = lookup[m.id]?.[t.id]
                  return (
                    <td key={t.id} className={e ? '' : 'muted'}>
                      {e ? e.score.toFixed(3) : '—'}
                    </td>
                  )
                })}
                <td>
                  <strong>{(normalizedAvg[m.id] ?? 0).toFixed(3)}</strong>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
