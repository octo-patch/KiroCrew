/**
 * A send that failed before the gateway took it hands the picture identities
 * back beside their chips, or the retry of a redacted picture goes out with a
 * list the gateway cannot resolve. Both failure paths of the composer send (the
 * gateway refusal and the session creation that failed first) go through this
 * one helper, so they cannot drift apart again.
 */
import { describe, expect, it } from 'vitest'
import { handBackImageIds, type ImageIdMap } from './chatImageIdDrafts'

const redacted = '/tmp/uploads/[REDACTED: credential].png'
const ids: ImageIdMap = { [redacted]: '1234:567:89abcdef01234567' }

describe('handBackImageIds', () => {
  it('merges the identities into the live map when the composer is on screen', () => {
    const live: ImageIdMap = { '/tmp/other.png': 'aaaa:1:bbbb' }
    const drafts: Record<string, ImageIdMap> = {}
    handBackImageIds(ids, true, live, drafts, 'slot-a')
    expect(live).toEqual({ '/tmp/other.png': 'aaaa:1:bbbb', ...ids })
    expect(drafts).toEqual({})
  })

  it('parks the identities on the slot draft when the composer is elsewhere, merging with what is parked', () => {
    const live: ImageIdMap = {}
    const drafts: Record<string, ImageIdMap> = { 'slot-a': { '/tmp/kept.png': 'cccc:2:dddd' } }
    handBackImageIds(ids, false, live, drafts, 'slot-a')
    expect(live).toEqual({})
    expect(drafts['slot-a']).toEqual({ '/tmp/kept.png': 'cccc:2:dddd', ...ids })
  })

  it('does nothing without identities to hand back', () => {
    const live: ImageIdMap = { '/tmp/other.png': 'aaaa:1:bbbb' }
    const drafts: Record<string, ImageIdMap> = {}
    handBackImageIds(undefined, true, live, drafts, 'slot-a')
    expect(live).toEqual({ '/tmp/other.png': 'aaaa:1:bbbb' })
    expect(drafts).toEqual({})
  })
})
