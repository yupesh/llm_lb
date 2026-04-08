import { useState } from 'react'
import type { Index } from '../types'
import { fmtCost, fmtScore } from '../format'

export function TasksView({ index }: { index: Index }) {
  const [open, setOpen] = useState<string | null>(index.tasks[0]?.id ?? null)

  return (
    <div className="list">
      {index.tasks.map((task) => {
        const ranking = index.matrix
          .filter((e) => e.task_id === task.id)
          .sort((a, b) => b.score - a.score || b.tps - a.tps)
        const isOpen = open === task.id
        return (
          <section key={task.id} className="card">
            <header onClick={() => setOpen(isOpen ? null : task.id)}>
              <h2>{task.id}</h2>
              <span className="version">v{task.version}</span>
              <span className="badge">{task.primary_metric}</span>
              <span className="meta">
                {task.n_samples} samples · {ranking.length} model(s)
              </span>
            </header>
            {isOpen && (
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Model</th>
                    <th>{task.primary_metric}</th>
                    <th>TPS</th>
                    <th>p95 (ms)</th>
                    <th>Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {ranking.map((e, i) => (
                    <tr key={e.model_id}>
                      <td>{i + 1}</td>
                      <td>
                        <code>{e.model_id}</code>
                      </td>
                      <td>{fmtScore(e.score)}</td>
                      <td>{e.tps.toFixed(1)}</td>
                      <td>{e.p95_latency_ms.toFixed(1)}</td>
                      <td>{fmtCost(e.cost_usd)}</td>
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
