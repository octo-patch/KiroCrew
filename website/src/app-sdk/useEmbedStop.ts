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
 * A soft press that lands while the session's automatic compaction holds it is
 * DECLINED: the reply is `{ok: true, compacting: true}`, nothing is stopping
 * (`stop_state` stays `idle`), and the backend arms its hatch so that the NEXT
 * press restarts the session, with or without `force`. So a declined press
 * drops the optimistic `soft_pending` (nothing is "Stopping") and arms
 * `stopDeclined` for the backend's window, during which the host warns that
 * the next press resets the session. That press is sent as a plain stop and
 * the backend decides: a hard kill while the compaction lasts, an ordinary
 * soft stop once it has ended. The window is timed from the reply, so it ends
 * no earlier than the server's: a hint that outlives the marker over-warns,
 * while one that ended first would let a press reset the session unannounced.
 * The slot read's `stop_declined` (the backend's own marker) arms the same
 * warning, so an embed opened or re-entered inside the window warns too.
 *
 * Both calls go to `POST /api/chat/slots/{slot}/stop[?force=true]`, which the
 * embed's existing `/api/chat` grant already covers.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { handleStopPress, isEscalationState } from '../utils/stopDebounce'
import { STOP_DECLINED_HINT_MS } from '../hooks/useStopDeclinedHint'
import { i18nT } from '../i18n/t'
import type { AppApi } from './scopedApi'

export type EmbedStopState = 'idle' | 'soft_pending' | 'killing'

type StopReply = { ok?: boolean; compacting?: boolean; error?: string; code?: string } | null | undefined

export function useEmbedStop({ api, slotKey, running, serverStopState, serverStopDeclined, onSettled }: {
  api: Pick<AppApi, 'post'>
  slotKey: string
  running: boolean
  serverStopState?: EmbedStopState
  /** The slot read's `stop_declined`: the backend's own decline marker. */
  serverStopDeclined?: boolean
  onSettled: () => void
}) {
  const softStopAtRef = useRef(0)
  const [optimistic, setOptimistic] = useState(false)
  const [error, setError] = useState('')
  // When the last press was declined for a compaction (0 when it was not):
  // the next press resets the session, and the host must say so first.
  const [declinedAt, setDeclinedAt] = useState(0)
  // Only the latest press's reply may arm or retire the decline: a decline
  // answered late must not re-arm the warning over the press that followed it.
  const pressSeqRef = useRef(0)
  // Which turn of which slot a press was made in. Bumped when the slot changes
  // and when the turn ends, so a decline answered after either is not armed
  // over a turn it was never about.
  const turnEpochRef = useRef(0)
  // Everything the protocol remembers is about ONE slot.
  const slotRef = useRef(slotKey)
  useEffect(() => {
    slotRef.current = slotKey
    turnEpochRef.current += 1
    softStopAtRef.current = 0
    setOptimistic(false)
    setDeclinedAt(0)
    setError('')
  }, [slotKey])
  useEffect(() => {
    if (!optimistic) return
    if (!running || isEscalationState(serverStopState)) setOptimistic(false)
  }, [optimistic, running, serverStopState])
  // A finished turn retires a stale failure notice with it, and the decline: a
  // turn cannot finish while the compaction holds its session, and the
  // compaction's end drops the backend's marker, so the next turn's first
  // press is an ordinary one.
  useEffect(() => {
    if (running) return
    turnEpochRef.current += 1
    setError('')
    setDeclinedAt(0)
  }, [running])
  // A stop the server reports under way supersedes the decline.
  useEffect(() => { if (isEscalationState(serverStopState)) setDeclinedAt(0) }, [serverStopState])
  // The backend's marker expires after its window; so does the warning.
  useEffect(() => {
    if (!declinedAt) return
    const timer = setTimeout(() => setDeclinedAt(0), STOP_DECLINED_HINT_MS)
    return () => clearTimeout(timer)
  }, [declinedAt])

  const stopState: EmbedStopState =
    optimistic && !isEscalationState(serverStopState) ? 'soft_pending' : (serverStopState ?? 'idle')
  // Armed by the decline reply at once (the poll lags it) or by the slot read's
  // own marker (an embed that opens or returns mid-window). The embed's own
  // optimistic soft_pending does not retire it: while the press after a decline
  // is unanswered it may still be the reset, so only the server's phase does.
  const stopDeclined = running && !isEscalationState(serverStopState) && (declinedAt > 0 || serverStopDeclined === true)

  const stop = useCallback((force: boolean) => {
    setError('')
    const sentFor = slotKey
    const seq = ++pressSeqRef.current
    const epoch = turnEpochRef.current
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
          return
        }
        if (seq !== pressSeqRef.current || epoch !== turnEpochRef.current) return
        if (reply?.compacting) {
          // Declined: nothing is stopping, and the next press is the reset.
          // softStopAtRef keeps the declined press's time, so a double-tap
          // still cannot spend the hatch inside the arming window.
          setOptimistic(false)
          setDeclinedAt(Date.now())
        } else {
          // The press landed (a stop, or the hatch's reset): the decline is spent.
          setDeclinedAt(0)
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
    const soft = () => { setOptimistic(true); stop(false) }
    handleStopPress(
      // After a decline the press is the hatch: debounced like a second press,
      // but sent as a plain stop, since the backend's marker is what escalates
      // it, and once the compaction has ended the same press is a soft stop.
      isEscalationState(stopState) || stopDeclined,
      Date.now(),
      softStopAtRef,
      soft,
      () => {
        if (!stopDeclined) { stop(true); return }
        softStopAtRef.current = Date.now()
        soft()
      },
    )
  }, [stop, stopState, stopDeclined])

  return { stopState, onStop, stopError: error, stopDeclined }
}
