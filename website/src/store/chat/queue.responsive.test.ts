import { describe, it, expect, vi, afterEach } from 'vitest'
import reducer, { appendSlotMessage, appendQueuedMessage, removeQueuedMessage, cancelQueuedMessage, markSendUnconfirmed, selectSendAccepted, selectSendConfirmed } from '../chatSlice'
import type { RootState } from '../index'
import { hydrateQueuedBubbles } from './queue'
import { normalizeQueueEntries } from './wire'

const SLOT = 'member-exact'
const TS = '2026-10-14T08:12:00Z'
afterEach(() => vi.useRealTimers())

function initial(active: boolean) {
  let state = { ...reducer(undefined, { type: '@@INIT' }), activeSlot: active ? SLOT : 'other' }
  state = reducer(state, appendSlotMessage({ slot: SLOT, message: { role: 'user', content: 'same text', cls: '', ts: TS, meta: { sendId: 'mine', responsiveSend: true } } }))
  return state
}
const rows = (state: ReturnType<typeof initial>) => state.activeSlot === SLOT ? state.messages : state.slotMessages[SLOT]
const root = (state: ReturnType<typeof initial>) => ({ chat: state }) as RootState
const push = { slot: SLOT, content: 'same text', ts: TS, queue_id: 'q1', meta: { sendId: 'mine', files: ['/work/My File.txt'], quote: { role: 'assistant', text: 'quoted' } } }
const pop = { slot: SLOT, content: 'same text', queue_id: 'q1', meta: push.meta }

describe.each([true, false])('exact queued-send identity (active=%s)', (active) => {
  it('queue acceptance replaces pending text without claiming execution; cancel remains available', () => {
    let state = initial(active)
    state = reducer(state, markSendUnconfirmed({ slot: SLOT, sendId: 'mine' }))
    state = reducer(state, appendQueuedMessage(push))
    expect(rows(state)).toHaveLength(1)
    expect(rows(state)[0]).toMatchObject({ role: 'queued', meta: { queueId: 'q1', ...push.meta } })
    expect(rows(state)[0].meta?.deliveryUnconfirmed).toBeUndefined()
    expect(selectSendAccepted(root(state), SLOT, 'mine')).toBe(true)
    expect(selectSendConfirmed(root(state), SLOT, 'mine')).toBe(false)
    state = reducer(state, cancelQueuedMessage({ slot: SLOT, queue_id: 'q1' }))
    expect(rows(state)).toHaveLength(0)
  })

  it('drain proves execution and duplicates or stale pushes do not append again', () => {
    let state = reducer(initial(active), appendQueuedMessage(push))
    state = reducer(state, removeQueuedMessage(pop))
    state = reducer(state, removeQueuedMessage(pop))
    state = reducer(state, appendQueuedMessage(push))
    expect(rows(state)).toHaveLength(1)
    expect(rows(state)[0]).toMatchObject({ role: 'user', meta: push.meta })
    expect(selectSendConfirmed(root(state), SLOT, 'mine')).toBe(true)
  })

  it('a pop can reconcile its exact provisional row even when push was missed', () => {
    const state = reducer(initial(active), removeQueuedMessage(pop))
    expect(rows(state)).toHaveLength(1)
    expect(rows(state)[0].role).toBe('user')
    expect(selectSendConfirmed(root(state), SLOT, 'mine')).toBe(true)
  })

  it('does not deduplicate identical text from a different send', () => {
    const state = reducer(initial(active), appendQueuedMessage({ ...push, meta: { sendId: 'other-send' } }))
    expect(rows(state)).toHaveLength(2)
    expect(selectSendAccepted(root(state), SLOT, 'mine')).toBe(false)
  })

  it('rehydrates the exact queue identity and preserves files and quote', () => {
    vi.setSystemTime(new Date(TS))
    const state = initial(active)
    const queue = normalizeQueueEntries([{ id: 'q1', content: push.content, meta: push.meta }])
    const hydrated = hydrateQueuedBubbles(rows(state), queue)
    expect(hydrated).toHaveLength(1)
    expect(hydrated[0]).toMatchObject({ role: 'queued', meta: { queueId: 'q1', ...push.meta } })
    expect(hydrateQueuedBubbles(hydrated, queue)).toEqual(hydrated)
  })
})

it.each([null, ['mine'], 42, '', 'x'.repeat(129), 'bad id'])('ignores malformed queue correlation %j', (sendId) => {
  const state = reducer(initial(true), appendQueuedMessage({ ...push, meta: { sendId } }))
  expect(rows(state)).toHaveLength(2)
  expect(rows(state)[1].meta?.sendId).toBeUndefined()
})
