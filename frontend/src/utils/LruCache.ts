/**
 * Simple LRU (Least Recently Used) cache implementation.
 * Evicts oldest entries when max size is exceeded.
 */
export class LruCache<K, V> {
  private cache = new Map<K, V>()
  private onEvict?: (value: V) => void

  constructor(private maxSize: number, onEvict?: (value: V) => void) {
    this.onEvict = onEvict
  }

  get(key: K): V | undefined {
    const value = this.cache.get(key)
    if (value !== undefined) {
      // Move to end (most recently used)
      this.cache.delete(key)
      this.cache.set(key, value)
    }
    return value
  }

  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      this.cache.delete(key)
    } else if (this.cache.size >= this.maxSize) {
      // Evict oldest (first key)
      const oldest = this.cache.keys().next().value
      if (oldest !== undefined) {
        const evicted = this.cache.get(oldest)
        if (evicted !== undefined && this.onEvict) {
          this.onEvict(evicted)
        }
        this.cache.delete(oldest)
      }
    }
    this.cache.set(key, value)
  }

  has(key: K): boolean {
    return this.cache.has(key)
  }

  delete(key: K): boolean {
    const value = this.cache.get(key)
    if (value !== undefined && this.onEvict) {
      this.onEvict(value)
    }
    return this.cache.delete(key)
  }

  clear(): void {
    if (this.onEvict) {
      for (const value of this.cache.values()) {
        this.onEvict(value)
      }
    }
    this.cache.clear()
  }

  get size(): number {
    return this.cache.size
  }
}
