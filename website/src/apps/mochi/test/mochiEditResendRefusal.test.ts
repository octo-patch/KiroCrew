/**
 * An edit-resend replaces the edited row and every row after it. Which rows
 * those are is read BEFORE the edit route is awaited: a reply the socket
 * appends while the request is in flight is the replacement's own answer and
 * must stay on screen. A refused edit of a message with pictures sends nothing,
 * so it removes nothing; the notice it shows says the message keeps its
 * pictures, and a transcript cut from under it would say the opposite.
 */
import { describe, expect, it } from 'vitest'
import { editResendOutcome, editSuffixIds, transcriptAfterEditResend } from '../src/renderer/ChatPanel'

const rows = [
  { id: 'a', role: 'user', timestamp: 1, content: 'first' },
  { id: 'b', role: 'assistant', timestamp: 2, content: 'reply' },
  { id: 'c', role: 'user', timestamp: 3, content: 'edit me' },
  { id: 'd', role: 'assistant', timestamp: 4, content: 'later reply' },
]
const arrivedMeanwhile = { id: 'e', role: 'assistant', timestamp: 5, content: 'the edit\'s own reply' }

describe('mochi edit-resend outcome', () => {
  it('a refused edit of a message with pictures sends nothing', () => {
    expect(editResendOutcome(false, true)).toBe('refused_untouched')
  })
  it('a refused text-only edit falls back to a plain send; an accepted edit is accepted', () => {
    expect(editResendOutcome(false, false)).toBe('fallback_send')
    expect(editResendOutcome(true, true)).toBe('accepted')
    expect(editResendOutcome(true, false)).toBe('accepted')
  })
})

describe('mochi edit suffix, captured before the edit route is awaited', () => {
  it('is the edited row and every row after it', () => {
    expect([...editSuffixIds(rows, 3)]).toEqual(['c', 'd'])
  })
  it('is empty for a row the panel does not hold', () => {
    expect(editSuffixIds(rows, 9).size).toBe(0)
  })
})

describe('mochi transcript after an edit-resend attempt', () => {
  const suffix = editSuffixIds(rows, 3)

  it('an accepted edit removes the captured suffix and keeps a reply that arrived during the request', () => {
    const during = [...rows, arrivedMeanwhile]
    expect(transcriptAfterEditResend(during, suffix, 'accepted')).toEqual([rows[0], rows[1], arrivedMeanwhile])
  })
  it('a text-only fallback send removes the captured suffix the same way', () => {
    expect(transcriptAfterEditResend(rows, suffix, 'fallback_send')).toEqual(rows.slice(0, 2))
  })
  it('a refused picture edit leaves the panel exactly as it was', () => {
    const during = [...rows, arrivedMeanwhile]
    expect(transcriptAfterEditResend(during, suffix, 'refused_untouched')).toBe(during)
  })
  it('an empty suffix removes nothing', () => {
    expect(transcriptAfterEditResend(rows, editSuffixIds(rows, 9), 'accepted')).toBe(rows)
  })
})
