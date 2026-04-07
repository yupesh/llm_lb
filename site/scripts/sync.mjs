// Copies the aggregated leaderboard index from ../data into site/public/ so
// Vite can serve it as a static asset both in `bun run dev` and in builds.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const src = resolve(here, '..', '..', 'data', 'index.json')
const dst = resolve(here, '..', 'public', 'data', 'index.json')

if (!existsSync(src)) {
  console.error(
    `[sync] missing ${src}\n` +
    `       run \`cd ../runner && uv run llm-lb aggregate --root ..\` first.`
  )
  process.exit(1)
}

mkdirSync(dirname(dst), { recursive: true })
copyFileSync(src, dst)
console.log(`[sync] ${src} -> ${dst}`)
