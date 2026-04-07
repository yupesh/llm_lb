import { useEffect, useState } from 'react'
import type { Index } from './types'
import { loadIndex } from './data'
import { TasksView } from './views/TasksView'
import { ModelsView } from './views/ModelsView'
import { CompareView } from './views/CompareView'

type View = 'tasks' | 'models' | 'compare'

export default function App() {
  const [index, setIndex] = useState<Index | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<View>('tasks')

  useEffect(() => {
    loadIndex().then(setIndex).catch((e) => setError(String(e)))
  }, [])

  if (error) return <div className="error">Error: {error}</div>
  if (!index) return <div className="loading">Loading…</div>

  return (
    <div className="app">
      <header className="app-header">
        <h1>LLM Leaderboard</h1>
        <nav>
          <button className={view === 'tasks' ? 'active' : ''} onClick={() => setView('tasks')}>
            Tasks ({index.tasks.length})
          </button>
          <button className={view === 'models' ? 'active' : ''} onClick={() => setView('models')}>
            Models ({index.models.length})
          </button>
          <button className={view === 'compare' ? 'active' : ''} onClick={() => setView('compare')}>
            Compare
          </button>
        </nav>
        <div className="meta">Updated: {new Date(index.generated_at).toLocaleString()}</div>
      </header>
      <main>
        {view === 'tasks' && <TasksView index={index} />}
        {view === 'models' && <ModelsView index={index} />}
        {view === 'compare' && <CompareView index={index} />}
      </main>
    </div>
  )
}
