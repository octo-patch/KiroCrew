/**
 * useEmbedStop — the failure and slot-switch branches the ChatEmbed render
 * tests do not reach. Each case is a way a Stop press could silently lie to
 * the user: a lost request that cleared the pending state (so the retry read
 * as a fresh soft press and the backend hard-killed the queue without
 * warning), a refusal that kept the optimistic `soft_pending`, or a reply for
 * the previous slot landing on the new one.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { useEmbedStop, type EmbedStopState } from '../app-sdk/useEmbedStop'
import { FORCE_KILL_ARMING_MS } from '../utils/stopDebounce'
import { STOP_DECLINED_HINT_MS } from '../hooks/useStopDeclinedHint'

type Props = { slotKey: string; running: boolean; serverStopState?: EmbedStopState; serverStopDeclined?: boolean }

function setup(post: (path: string, body: unknown) => Promise<unknown>, initial: Props) {
  const onSettled = vi.fn()
  const api = { post: vi.fn(post) } as unknown as { post: <T>(p: string, b: unknown) => Promise<T> }
  const hook = renderHook((p: Props) => useEmbedStop({ api, onSettled, ...p }), { initialProps: initial })
  return { hook, api: api as unknown as { post: ReturnType<typeof vi.fn> }, onSettled }
}

describe('useEmbedStop', () => {
  it('keeps the pending state on a request that never got an answer', async () => {
    const { hook, onSettled } = setup(() => Promise.reject(new TypeError('Failed to fetch')), {
      slotKey: 's1', running: true,
    })
    act(() => hook.result.current.onStop())
    await waitFor(() => expect(onSettled).toHaveBeenCalled())
    // It may have armed the cancel, so the retry must be a deliberate force.
    expect(hook.result.current.stopState).toBe('soft_pending')
    expect(hook.result.current.stopError).not.toBe('')
  })

  it('clears the pending state on an HTTP error so the retry is a fresh soft press', async () => {
    const err = Object.assign(new Error('boom'), { status: 500 })
    const { hook, onSettled } = setup(() => Promise.reject(err), { slotKey: 's1', running: true })
    act(() => hook.result.current.onStop())
    await waitFor(() => expect(onSettled).toHaveBeenCalled())
    expect(hook.result.current.stopState).toBe('idle')
    expect(hook.result.current.stopError).toContain('boom')
  })

  it('drops a reply that arrives after the embed moved to another slot', async () => {
    let reject!: (e: unknown) => void
    const { hook, onSettled } = setup(
      () => new Promise((_, r) => { reject = r }),
      { slotKey: 's1', running: true },
    )
    act(() => hook.result.current.onStop())
    hook.rerender({ slotKey: 's2', running: true })
    await act(async () => { reject(Object.assign(new Error('late'), { status: 500 })) })
    await waitFor(() => expect(onSettled).toHaveBeenCalled())
    expect(hook.result.current.stopError).toBe('')
    expect(hook.result.current.stopState).toBe('idle')
  })

  it('drops a refusal that arrives after the embed moved to another slot', async () => {
    let resolve!: (v: unknown) => void
    const { hook, onSettled } = setup(
      () => new Promise((r) => { resolve = r }),
      { slotKey: 's1', running: true },
    )
    act(() => hook.result.current.onStop())
    hook.rerender({ slotKey: 's2', running: true })
    await act(async () => { resolve({ ok: false, error: 'late refusal' }) })
    await waitFor(() => expect(onSettled).toHaveBeenCalled())
    expect(hook.result.current.stopError).toBe('')
  })

  it('retires the optimistic phase when the turn ends or the server reports its own', async () => {
    const { hook } = setup(() => Promise.resolve({ ok: true }), { slotKey: 's1', running: true })
    act(() => hook.result.current.onStop())
    expect(hook.result.current.stopState).toBe('soft_pending')
    hook.rerender({ slotKey: 's1', running: true, serverStopState: 'killing' })
    expect(hook.result.current.stopState).toBe('killing')
    hook.rerender({ slotKey: 's1', running: false })
    expect(hook.result.current.stopState).toBe('idle')
  })

  it('encodes the slot key in the stop path', async () => {
    const { hook, api, onSettled } = setup(() => Promise.resolve({ ok: true }), { slotKey: 'a/b c', running: true })
    act(() => hook.result.current.onStop())
    expect(api.post).toHaveBeenCalledWith('/api/chat/slots/a%2Fb%20c/stop', {})
    // The reply settles the decline flag; let it land inside act.
    await waitFor(() => expect(onSettled).toHaveBeenCalled())
  })
})

// A press during the session's automatic compaction is DECLINED by the backend
// (`{ok: true, compacting: true}`, stop_state left idle) and arms its hatch: the
// next press, force or not, restarts the session. Fake timers drive both the
// double-tap window and the decline window.
describe('useEmbedStop after a compaction decline', () => {
  const DECLINED = { ok: true, info: 'compacting', compacting: true }
  const paths = (api: { post: ReturnType<typeof vi.fn> }) => api.post.mock.calls.map(([p]) => p)
  const flush = () => act(async () => {})

  beforeEach(() => { vi.useFakeTimers(); vi.setSystemTime(new Date('2026-01-01T00:00:00Z')) })
  afterEach(() => { vi.useRealTimers() })

  async function declined(post?: (path: string) => Promise<unknown>) {
    let n = 0
    const ctx = setup(
      (path) => (n++ === 0 ? Promise.resolve(DECLINED) : (post ? post(path) : Promise.resolve({ ok: true }))),
      { slotKey: 's1', running: true },
    )
    act(() => ctx.hook.result.current.onStop())
    await flush()
    return ctx
  }

  it('drops the optimistic Stopping and arms the reset warning', async () => {
    const { hook } = await declined()
    expect(hook.result.current.stopState).toBe('idle')
    expect(hook.result.current.stopDeclined).toBe(true)
  })

  it('sends the next press as a plain stop and spends the decline when it lands', async () => {
    const { hook, api } = await declined()
    act(() => { vi.advanceTimersByTime(FORCE_KILL_ARMING_MS + 1) })
    act(() => hook.result.current.onStop())
    // The backend's marker escalates it; while compacting it is the reset,
    // once the compaction ended the same press is an ordinary soft stop.
    expect(paths(api)).toEqual(['/api/chat/slots/s1/stop', '/api/chat/slots/s1/stop'])
    expect(hook.result.current.stopState).toBe('soft_pending')
    await flush()
    expect(hook.result.current.stopDeclined).toBe(false)
  })

  it('ignores a double-tap on the declined press inside the arming window', async () => {
    const { hook, api } = await declined()
    act(() => hook.result.current.onStop())
    expect(paths(api)).toHaveLength(1)
    expect(hook.result.current.stopDeclined).toBe(true)
  })

  it('retires the warning with the backend window, so the next press is a fresh soft press', async () => {
    const { hook } = await declined()
    act(() => { vi.advanceTimersByTime(STOP_DECLINED_HINT_MS) })
    expect(hook.result.current.stopDeclined).toBe(false)
  })

  it('retires the warning when the turn ends or the embed changes slot', async () => {
    const a = await declined()
    a.hook.rerender({ slotKey: 's1', running: false })
    expect(a.hook.result.current.stopDeclined).toBe(false)
    const b = await declined()
    b.hook.rerender({ slotKey: 's2', running: true })
    expect(b.hook.result.current.stopDeclined).toBe(false)
  })

  it('lets a stop the server reports supersede the decline', async () => {
    const { hook } = await declined()
    // A stop from another surface: the poll shows a phase of its own, and the
    // backend's escalation consumed the marker, so idle again is not armed.
    hook.rerender({ slotKey: 's1', running: true, serverStopState: 'soft_pending' })
    hook.rerender({ slotKey: 's1', running: true, serverStopState: 'idle' })
    expect(hook.result.current.stopDeclined).toBe(false)
  })

  it('keeps the warning when the reset press fails outright, since the hatch is still armed', async () => {
    const { hook } = await declined(() => Promise.reject(Object.assign(new Error('boom'), { status: 500 })))
    act(() => { vi.advanceTimersByTime(FORCE_KILL_ARMING_MS + 1) })
    act(() => hook.result.current.onStop())
    await flush()
    expect(hook.result.current.stopError).toContain('boom')
    expect(hook.result.current.stopDeclined).toBe(true)
  })

  it('keeps the reset warning while the reset press has no answer, so the retry is still announced', async () => {
    const { hook, api } = await declined(() => Promise.reject(new TypeError('Failed to fetch')))
    act(() => { vi.advanceTimersByTime(FORCE_KILL_ARMING_MS + 1) })
    act(() => hook.result.current.onStop())
    await flush()
    // The press may or may not have reached the backend; either way the next
    // press can be the reset, and the host must still say so.
    expect(hook.result.current.stopError).not.toBe('')
    expect(hook.result.current.stopDeclined).toBe(true)
    act(() => { vi.advanceTimersByTime(FORCE_KILL_ARMING_MS + 1) })
    act(() => hook.result.current.onStop())
    // Still a plain stop: the backend's marker decides whether it is the reset.
    expect(paths(api)).toEqual(['/api/chat/slots/s1/stop', '/api/chat/slots/s1/stop', '/api/chat/slots/s1/stop'])
    await flush()
  })

  // The slot-detail read reports the backend's own marker, so an embed that
  // opens (or returns) mid-window warns before its first press too.
  it('honours a decline the server reports when the embed opens', async () => {
    const { hook, api } = setup(() => Promise.resolve({ ok: true }), { slotKey: 's1', running: true, serverStopDeclined: true })
    expect(hook.result.current.stopDeclined).toBe(true)
    act(() => hook.result.current.onStop())
    expect(paths(api)).toEqual(['/api/chat/slots/s1/stop'])
    await flush()
  })

  it('stops warning once the server reports the marker expired or cleared', async () => {
    const { hook } = setup(() => Promise.resolve({ ok: true }), { slotKey: 's1', running: true, serverStopDeclined: true })
    hook.rerender({ slotKey: 's1', running: true, serverStopDeclined: false })
    expect(hook.result.current.stopDeclined).toBe(false)
  })

  it('does not warn on an armed server marker once a stop is under way', async () => {
    const { hook } = setup(() => Promise.resolve({ ok: true }), { slotKey: 's1', running: true, serverStopDeclined: true, serverStopState: 'killing' })
    expect(hook.result.current.stopDeclined).toBe(false)
  })

  it('does not let a late decline re-arm the warning over the press that followed it', async () => {
    let answerFirst!: (v: unknown) => void
    let n = 0
    const { hook } = setup(
      () => (n++ === 0 ? new Promise((r) => { answerFirst = r }) : Promise.resolve({ ok: true })),
      { slotKey: 's1', running: true },
    )
    act(() => hook.result.current.onStop())
    act(() => { vi.advanceTimersByTime(FORCE_KILL_ARMING_MS + 1) })
    act(() => hook.result.current.onStop())
    await flush()
    await act(async () => { answerFirst(DECLINED) })
    expect(hook.result.current.stopDeclined).toBe(false)
  })

  // A decline answered after its turn ended must not warn on the next turn: the
  // compaction that declined it is over, and the backend dropped its marker.
  it('does not arm a decline that is answered after its turn ended', async () => {
    let answer!: (v: unknown) => void
    const { hook } = setup(() => new Promise((r) => { answer = r }), { slotKey: 's1', running: true })
    act(() => hook.result.current.onStop())
    hook.rerender({ slotKey: 's1', running: false })
    hook.rerender({ slotKey: 's1', running: true })
    await act(async () => { answer(DECLINED) })
    expect(hook.result.current.stopDeclined).toBe(false)
  })

  it('does not arm a decline that is answered after the embed left the slot and came back', async () => {
    let answer!: (v: unknown) => void
    const { hook } = setup(() => new Promise((r) => { answer = r }), { slotKey: 's1', running: true })
    act(() => hook.result.current.onStop())
    hook.rerender({ slotKey: 's2', running: true })
    hook.rerender({ slotKey: 's1', running: true })
    await act(async () => { answer(DECLINED) })
    expect(hook.result.current.stopDeclined).toBe(false)
  })
})
