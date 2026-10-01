/**
 * Per-slot persistence of the identity the gateway recorded for each staged
 * picture chip (`meta.image_ids`, keyed by the chip's path). Thin instance of
 * `createSlotDraftStore`, stored beside `chatFileDrafts` with the same storage
 * and lifetime: sessionStorage, no TTL, no cap.
 *
 * A redacted picture path names no file, so the identity is what lets a
 * resend recover the picture. Kept per slot like the chips themselves: a send
 * in one slot must not drop the identities another slot's restored chips
 * still need, and a slot switch must bring them back with the chips.
 */
import { createSlotDraftStore } from './slotDraftStore'

export const IMAGE_ID_DRAFTS_KEY = 'mc-chat-image-id-drafts'

export type ImageIdMap = Record<string, string>

/** Keep only `path -> identity` entries with a non-blank string identity;
 *  return a fresh copy, or null when nothing is left. */
const sanitizeIds = (v: unknown): ImageIdMap | null => {
  if (!v || typeof v !== 'object' || Array.isArray(v)) return null
  const out: ImageIdMap = {}
  for (const [path, id] of Object.entries(v as Record<string, unknown>)) {
    if (typeof id === 'string' && id.trim() && id.length <= 64) out[path] = id
  }
  return Object.keys(out).length ? out : null
}

const store = createSlotDraftStore<ImageIdMap>({
  key: IMAGE_ID_DRAFTS_KEY,
  storage: 'session',
  sanitize: sanitizeIds,
})

export const loadImageIdDrafts = store.load
export const saveImageIdDrafts = store.save
export const setImageIdDraft = store.set

/** Hand a refused send's picture identities back beside their chips: into the
 *  live map when the composer is on screen, else into the slot's parked draft.
 *  Merged, never replaced, so a picture restored meanwhile keeps its own. */
export function handBackImageIds(
  imageIdsAtSend: ImageIdMap | undefined,
  onScreen: boolean,
  live: ImageIdMap,
  drafts: Record<string, ImageIdMap>,
  slot: string,
): void {
  if (!imageIdsAtSend) return
  if (onScreen) Object.assign(live, imageIdsAtSend)
  else setImageIdDraft(drafts, slot, { ...(drafts[slot] ?? {}), ...imageIdsAtSend })
}
