import { describe, expect, it } from 'vitest'

const jsdomWindow = () => (globalThis as typeof globalThis & { jsdom: { window: Window } }).jsdom.window

describe('Vitest browser storage environment', () => {
  it('uses the active JSDOM localStorage implementation', () => {
    expect(window.localStorage).toBe(jsdomWindow().localStorage)
  })

  it('uses the active JSDOM sessionStorage implementation', () => {
    expect(window.sessionStorage).toBe(jsdomWindow().sessionStorage)
  })
})
