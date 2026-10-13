/**
 * Stop for ChatEmbed: the press protocol ChatPane and ChatPage use, over the
 * embed's own scoped API instead of the Redux `requestStop` thunk (the embed
 * has no store).
 *
 * The first press is the cooperative soft cancel; a press while the slot is
 * `soft_pending` (or a stalled `killing`) escalates to the hard kill, and a
 * double-tap inside the arming window is ignored — `handleStopPress` is the
 * shared decision. The embed learns the stop phase only from its poll
 * (`stop_state` on the slot-detail read), which lags the press by up to a poll,
 * so its own soft press is held as an OPTIMISTIC `soft_pending` until the
 * server reports a phase of its own or the turn ends. Without that, a second
 * press inside the lag would read as a fresh soft press, and the backend
 * escalates any second press to a queue-clearing hard kill — silently.
 *
 * Both calls go to `POST /api/chat/slots/{slot}/stop[?force=true]`, which the
 * embed's existing `/api/chat` grant already covers.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { Loader2, Square } from 'lucide-react'
import { motion } from 'framer-motion'
import { handleStopPress, isEscalationState } from '../utils/stopDebounce'
import { i18nT } from '../i18n/t'
import type { AppApi } from './scopedApi'

export type EmbedStopState = 'idle' | 'soft_pending' | 'killing'

type StopReply = { ok?: boolean; error?: string; code?: string } | null | undefined

export function useEmbedStop({ api, slotKey, running, serverStopState, onSettled }: {
  api: Pick<AppApi, 'post'>
  slotKey: string
  running: boolean
  serverStopState?: EmbedStopState
  onSettled: () => void
}) {
  const softStopAtRef = useRef(0)
  const [optimistic, setOptimistic] = useState(false)
  const [error, setError] = useState('')
  // Everything the protocol remembers is about ONE slot.
  const slotRef = useRef(slotKey)
  useEffect(() => {
    slotRef.current = slotKey
    softStopAtRef.current = 0
    setOptimistic(false)
    setError('')
  }, [slotKey])
  useEffect(() => {
    if (!optimistic) return
    if (!running || isEscalationState(serverStopState)) setOptimistic(false)
  }, [optimistic, running, serverStopState])
  // A finished turn retires a stale failure notice with it.
  useEffect(() => { if (!running) setError('') }, [running])

  const stopState: EmbedStopState =
    optimistic && !isEscalationState(serverStopState) ? 'soft_pending' : (serverStopState ?? 'idle')

  const stop = useCallback((force: boolean) => {
    setError('')
    const sentFor = slotKey
    const path = `/api/chat/slots/${encodeURIComponent(slotKey)}/stop${force ? '?force=true' : ''}`
    void api.post<StopReply>(path, {}).then(
      (reply) => {
        if (slotRef.current !== sentFor) return
        // A 2xx can still carry a refusal (a peer-bound slot whose crew could
        // not be reached). It armed nothing, so the retry is a fresh press.
        if (reply && reply.ok === false) {
          softStopAtRef.current = 0
          setOptimistic(false)
          setError(i18nT('components.chatPane.stop_failed', { error: reply.error || reply.code || 'stop refused' }))
        }
      },
      (err: unknown) => {
        if (slotRef.current !== sentFor) return
        const status = (err as { status?: unknown })?.status
        // No HTTP status means the request never got an answer: it may have
        // reached the backend and armed the cancel, so the pending state stays
        // and the retry is a deliberate force (ChatPane's GPT round 8 rule).
        if (typeof status !== 'number') {
          setError(i18nT('components.chatPane.stop_failed_network'))
          return
        }
        softStopAtRef.current = 0
        setOptimistic(false)
        setError(i18nT('components.chatPane.stop_failed', { error: err instanceof Error ? err.message : String(err) }))
      },
    ).finally(onSettled)
  }, [api, slotKey, onSettled])

  const onStop = useCallback(() => {
    handleStopPress(
      isEscalationState(stopState),
      Date.now(),
      softStopAtRef,
      () => { setOptimistic(true); stop(false) },
      () => stop(true),
    )
  }, [stop, stopState])

  return { stopState, onStop, stopError: error }
}

/** The Stop control, drawn where the embed's send button sits. */
export function EmbedStopButton({ stopState, onStop }: { stopState: EmbedStopState; onStop: () => void }) {
  if (stopState === 'killing') {
    return (
      <button
        type="button"
        className="p-2 rounded-md bg-danger text-danger-fg hover:opacity-80 transition-opacity"
        onClick={onStop}
        title={i18nT('components.chatInput.killing')}
        aria-label={i18nT('components.chatInput.killing_session')}
        data-testid="chat-embed-stop-killing"
      >
        <Loader2 size={16} className="animate-spin" />
      </button>
    )
  }
  if (stopState === 'soft_pending') {
    return (
      <motion.button
        type="button"
        className="p-2 rounded-md bg-danger/10 text-danger hover:bg-danger/20 transition-colors"
        onClick={onStop}
        title={i18nT('components.chatInput.force_kill_discards_in_progress_work_and_queued')}
        aria-label={i18nT('components.chatInput.force_kill_session_discards_in_progress_work_and')}
        animate={{ opacity: [0.8, 1, 0.8] }}
        transition={{ duration: 1.2, repeat: Infinity }}
        data-testid="chat-embed-stop-pending"
      >
        <Square size={16} fill="currentColor" />
      </motion.button>
    )
  }
  return (
    <button
      type="button"
      className="p-2 rounded-md text-danger hover:bg-danger/10 transition-colors"
      onClick={onStop}
      title={i18nT('components.chatInput.stop_generation')}
      aria-label={i18nT('components.chatInput.stop_generation')}
      data-testid="chat-embed-stop"
    >
      <Square size={16} fill="currentColor" />
    </button>
  )
}
