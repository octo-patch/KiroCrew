import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act, fireEvent } from '@testing-library/react'
import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { FORCE_KILL_ARMING_MS } from '../utils/stopDebounce'
import { KILLING_ESCAPE_MS } from '../hooks/useStopEscapeHatch'

// Stop in the embed: drawn where Send sits while a turn runs, sending the slot
// stop endpoint with the shared soft-then-force press protocol.
const mockGet = vi.fn()
const mockPost = vi.fn()

vi.mock('../app-sdk/index', () => ({
  useAppApi: () => ({ get: mockGet, post: mockPost }),
  useAppInfo: () => ({ name: 'test-app', version: '0', permissions: { api: ['/api/chat'], events: [] } }),
}))

vi.mock('../app-sdk/ChatMessageList', () => ({
  default: ({ transcript }: { transcript?: { belowRows?: React.ReactNode } }) => (
    <div data-testid="chat-message-list">{transcript?.belowRows}</div>
  ),
}))

import ChatEmbed from '../app-sdk/ChatEmbed'

let queryClient: QueryClient

async function mount() {
  await act(async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ChatEmbed slotKey="slot-1" />
      </QueryClientProvider>,
    )
  })
  await act(async () => { vi.advanceTimersByTime(100) })
}

const stopCalls = () => mockPost.mock.calls.filter(([p]) => String(p).includes('/stop'))

beforeEach(() => {
  vi.clearAllMocks()
  vi.useFakeTimers()
  mockPost.mockResolvedValue({ ok: true })
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
})

afterEach(() => {
  vi.useRealTimers()
})

describe('ChatEmbed Stop', () => {
  it('shows Stop instead of Send while a turn runs and the draft is empty', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '' })
    await mount()
    expect(screen.getByTestId('stop-button-armed')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Send message' })).toBeNull()
    // A draft mid-turn still sends (it queues), so the send control comes back,
    // named for what it does now (main's ComposerSend: "Queue message").
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'more' } })
    expect(screen.queryByTestId('stop-button-armed')).toBeNull()
    expect(screen.getByRole('button', { name: 'Queue message' })).toBeInTheDocument()
  })

  it('is not the guide\'s chat-page Stop target', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '' })
    await mount()
    // composer.stop lives on the chat page; an embed inside an app must not
    // register a second target the guide could point at.
    expect(screen.getByTestId('stop-button-armed')).not.toHaveAttribute('data-ui-location')
  })

  it('hides the press hint when a draft swaps Stop for Send', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '', stopping: true, stop_state: 'soft_pending' })
    await mount()
    expect(screen.getByTestId('chat-embed-stop-hint')).toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'more' } })
    // Send queues the draft; "Click again to force stop" would be a lie.
    expect(screen.queryByTestId('chat-embed-stop-hint')).toBeNull()
  })

  it('shows Send, not Stop, once the turn is over', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'assistant', content: 'done', ts: 1 }], running: false, title: '' })
    await mount()
    expect(screen.queryByTestId('stop-button-armed')).toBeNull()
    expect(screen.getByRole('button', { name: 'Send message' })).toBeInTheDocument()
  })

  it('soft-stops on the first press and force-stops on a deliberate second press', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '', stop_state: 'idle' })
    await mount()
    await act(async () => { fireEvent.click(screen.getByTestId('stop-button-armed')) })
    expect(stopCalls().map(([p]) => p)).toEqual(['/api/chat/slots/slot-1/stop'])
    // The poll has not reported soft_pending yet; the press is held optimistically.
    expect(screen.getByTestId('stop-button-pulsing')).toBeInTheDocument()
    expect(screen.getByTestId('chat-embed-stop-hint')).toHaveTextContent('Click again to force stop')
    // The cost of the force stop is shown where the user decides, not only on hover.
    expect(screen.getByTestId('chat-embed-stop-hint')).toHaveTextContent('discards in-progress work and queued messages')
    // One status while stopping: the placeholder no longer says the agent is working.
    expect(screen.getByRole('textbox')).toHaveAttribute('placeholder', 'Stopping…')
    // A double-tap inside the arming window does nothing.
    await act(async () => { fireEvent.click(screen.getByTestId('stop-button-pulsing')) })
    expect(stopCalls()).toHaveLength(1)
    await act(async () => { vi.advanceTimersByTime(FORCE_KILL_ARMING_MS + 50) })
    await act(async () => { fireEvent.click(screen.getByTestId('stop-button-pulsing')) })
    expect(stopCalls().map(([p]) => p)).toEqual([
      '/api/chat/slots/slot-1/stop',
      '/api/chat/slots/slot-1/stop?force=true',
    ])
  })

  it('reads the backend stop phase for the button and the working indicator', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '', stopping: true, stop_state: 'soft_pending' })
    await mount()
    expect(screen.getByTestId('stop-button-pulsing')).toBeInTheDocument()
    expect(screen.getByTestId('chat-footer')).toHaveTextContent(/Stopping/)
  })

  it('draws the main composer\'s killing spinner, then its escape hatch that force-stops again', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '', stopping: true, stop_state: 'killing' })
    await mount()
    expect(screen.getByTestId('stop-button-killing')).toBeDisabled()
    // Header and placeholder follow the footer's "Killing…".
    expect(screen.getByRole('textbox')).toHaveAttribute('placeholder', 'Killing…')
    expect(screen.queryByTestId('chat-embed-stop-hint')).toBeNull()
    await act(async () => { vi.advanceTimersByTime(KILLING_ESCAPE_MS + 100) })
    // The escape hatch's hint sits above the composer, not beside the button.
    // An instruction, not a status line.
    expect(screen.getByTestId('chat-embed-stop-hint')).toHaveTextContent('Click again to force stop · taking longer than expected')
    expect(screen.queryByTestId('stop-escape-hint')).toBeNull()
    await act(async () => { fireEvent.click(screen.getByTestId('stop-button-escape-hatch')) })
    expect(stopCalls().map(([p]) => p)).toEqual(['/api/chat/slots/slot-1/stop?force=true'])
  })

  it('says so when the stop is refused, and the retry is a fresh soft press', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '', stop_state: 'idle' })
    mockPost.mockResolvedValueOnce({ ok: false, error: 'crew unreachable' })
    await mount()
    await act(async () => { fireEvent.click(screen.getByTestId('stop-button-armed')) })
    expect(screen.getByTestId('chat-embed-stop-error')).toHaveTextContent('crew unreachable')
    expect(screen.getByTestId('stop-button-armed')).toBeInTheDocument()
  })

  it('after a compaction decline, reads as not stopping and warns that the next press resets the session', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '', stop_state: 'idle' })
    mockPost.mockResolvedValueOnce({ ok: true, info: 'compacting', compacting: true })
    await mount()
    await act(async () => { fireEvent.click(screen.getByTestId('stop-button-armed')) })
    // Nothing is stopping: no pulsing force-stop, no "Stopping…" anywhere.
    expect(screen.queryByTestId('stop-button-pulsing')).toBeNull()
    expect(screen.getByTestId('stop-button-armed')).toBeInTheDocument()
    expect(screen.getByRole('textbox').getAttribute('placeholder')).toMatch(/^Agent is working/)
    expect(screen.queryByText('Stopping…')).toBeNull()
    expect(screen.getByTestId('chat-embed-stop-hint')).toHaveTextContent('Click again to force stop (restarts the session')
    // The reset press is a plain stop: the backend's hatch is what escalates it.
    await act(async () => { vi.advanceTimersByTime(FORCE_KILL_ARMING_MS + 50) })
    await act(async () => { fireEvent.click(screen.getByTestId('stop-button-armed')) })
    expect(stopCalls().map(([p]) => p)).toEqual(['/api/chat/slots/slot-1/stop', '/api/chat/slots/slot-1/stop'])
  })

  it('warns before the first press when the slot read reports an armed decline', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '', stop_state: 'idle', stop_declined: true })
    await mount()
    expect(screen.getByTestId('chat-embed-stop-hint')).toHaveTextContent('Click again to force stop (restarts the session')
    await act(async () => { fireEvent.click(screen.getByTestId('stop-button-armed')) })
    expect(stopCalls().map(([p]) => p)).toEqual(['/api/chat/slots/slot-1/stop'])
  })

  it('keeps the reset warning beside the error when the press after a decline fails', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '', stop_state: 'idle' })
    mockPost
      .mockResolvedValueOnce({ ok: true, info: 'compacting', compacting: true })
      .mockResolvedValueOnce({ ok: false, error: 'crew unreachable' })
    await mount()
    await act(async () => { fireEvent.click(screen.getByTestId('stop-button-armed')) })
    await act(async () => { vi.advanceTimersByTime(FORCE_KILL_ARMING_MS + 50) })
    await act(async () => { fireEvent.click(screen.getByTestId('stop-button-armed')) })
    // The refusal armed nothing and spent nothing: the backend's hatch is still
    // armed, so the next press still restarts the session and must say so.
    expect(screen.getByTestId('chat-embed-stop-error')).toHaveTextContent('crew unreachable')
    expect(screen.getByTestId('chat-embed-stop-hint')).toHaveTextContent('Click again to force stop (restarts the session')
  })
})
