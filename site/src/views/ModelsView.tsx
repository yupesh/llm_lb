import { useState } from 'react'
import type { Index } from '../types'

export function ModelsView({ index }: { index: Index }) {
  const [open, setOpen] = useState<string | null>(index.models[0]?.id ?? null)

  return (
    <div className="list">
      {index.models.map((m) => {
        const entries = index.matrix.filter((e) => e.model_id === m.id)
        const isOpen = open === m.id
        return (
          <section key={m.id} className="card">
            <header onClick={() => setOpen(isOpen ? null : m.id)}>
              <h2>{m.display_name}</h2>
              <code>{m.id}</code>
              <span className="badge">{m.provider}</span>
              {m.hardware && (
                <span className="meta">
                  {m.hardware.gpu_count}× {m.hardware.gpu_type}
                  {m.hardware.quantization ? ` · ${m.hardware.quantization}` : ''}
                </span>
              )}
              <span className="meta">{entries.length} task(s)</span>
            </header>
            {isOpen && (
              <table>
                <thead>
                  <tr>
                    <th>Task</th>
                    <th>Score</th>
                    <th>TPS</th>
                    <th>p95 (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.length === 0 && (
                    <tr>
                      <td colSpan={4} className="muted">
                        No results yet
                      </td>
                    </tr>
                  )}
                  {entries.map((e) => (
                    <tr key={e.task_id}>
                      <td>{e.task_id}</td>
                      <td>{e.score.toFixed(4)}</td>
                      <td>{e.tps.toFixed(1)}</td>
                      <td>{e.p95_latency_ms.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        )
      })}
    </div>
  )
}
