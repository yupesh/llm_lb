// Copies aggregated leaderboard data from ../data into site/public/ so Vite
// can serve it as a static asset both in `bun run dev` and in builds.
// - data/index.json   → consumed by the SPA at runtime
// - data/feed.xml     → Atom feed exposed at /data/feed.xml for RSS readers
import { copyFileSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const dataDir = resolve(here, '..', '..', 'data')
const publicData = resolve(here, '..', 'public', 'data')

const required = ['index.json']
const optional = ['feed.xml']

for (const name of required) {
  const src = resolve(dataDir, name)
  if (!existsSync(src)) {
    console.error(
      `[sync] missing ${src}\n` +
      `       run \`cd ../runner && uv run llm-lb aggregate --root ..\` first.`
    )
    process.exit(1)
  }
}

mkdirSync(publicData, { recursive: true })
for (const name of [...required, ...optional]) {
  const src = resolve(dataDir, name)
  if (!existsSync(src)) continue
  const dst = resolve(publicData, name)
  copyFileSync(src, dst)
  console.log(`[sync] ${src} -> ${dst}`)
}
