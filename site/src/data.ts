import type { Index } from './types'

let cache: Promise<Index> | null = null

export function loadIndex(): Promise<Index> {
  if (!cache) {
    const url = new URL('data/index.json', document.baseURI).toString()
    cache = fetch(url).then((r) => {
      if (!r.ok) throw new Error(`Failed to load ${url}: ${r.status}`)
      return r.json() as Promise<Index>
    })
  }
  return cache
}
