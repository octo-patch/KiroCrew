import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act, fireEvent } from '@testing-library/react'
import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { FORCE_KILL_ARMING_MS } from '../utils/stopDebounce'

// Stop in the embed: drawn where Send sits while a turn runs, sending the slot
// stop endpoint with the shared soft-then-force press protocol.
const mockGet = vi.fn()
const mockPost = vi.fn()

vi.mock('../app-sdk/index', () => ({
  useAppApi: () => ({ get: mockGet, post: mockPost }),
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
    expect(screen.getByTestId('chat-embed-stop')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Send message' })).toBeNull()
    // A draft mid-turn still sends (it queues), so Send comes back.
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'more' } })
    expect(screen.queryByTestId('chat-embed-stop')).toBeNull()
    expect(screen.getByRole('button', { name: 'Send message' })).toBeInTheDocument()
  })

  it('shows Send, not Stop, once the turn is over', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'assistant', content: 'done', ts: 1 }], running: false, title: '' })
    await mount()
    expect(screen.queryByTestId('chat-embed-stop')).toBeNull()
    expect(screen.getByRole('button', { name: 'Send message' })).toBeInTheDocument()
  })

  it('soft-stops on the first press and force-stops on a deliberate second press', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '', stop_state: 'idle' })
    await mount()
    await act(async () => { fireEvent.click(screen.getByTestId('chat-embed-stop')) })
    expect(stopCalls().map(([p]) => p)).toEqual(['/api/chat/slots/slot-1/stop'])
    // The poll has not reported soft_pending yet; the press is held optimistically.
    expect(screen.getByTestId('chat-embed-stop-pending')).toBeInTheDocument()
    expect(screen.getByTestId('chat-embed-stop-hint')).toHaveTextContent('Click again to force stop')
    // A double-tap inside the arming window does nothing.
    await act(async () => { fireEvent.click(screen.getByTestId('chat-embed-stop-pending')) })
    expect(stopCalls()).toHaveLength(1)
    await act(async () => { vi.advanceTimersByTime(FORCE_KILL_ARMING_MS + 50) })
    await act(async () => { fireEvent.click(screen.getByTestId('chat-embed-stop-pending')) })
    expect(stopCalls().map(([p]) => p)).toEqual([
      '/api/chat/slots/slot-1/stop',
      '/api/chat/slots/slot-1/stop?force=true',
    ])
  })

  it('reads the backend stop phase for the button and the working indicator', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '', stopping: true, stop_state: 'soft_pending' })
    await mount()
    expect(screen.getByTestId('chat-embed-stop-pending')).toBeInTheDocument()
    expect(screen.getByTestId('chat-footer')).toHaveTextContent(/Stopping/)
  })

  it('says so when the stop is refused, and the retry is a fresh soft press', async () => {
    mockGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hi', ts: 1 }], running: true, title: '', stop_state: 'idle' })
    mockPost.mockResolvedValueOnce({ ok: false, error: 'crew unreachable' })
    await mount()
    await act(async () => { fireEvent.click(screen.getByTestId('chat-embed-stop')) })
    expect(screen.getByTestId('chat-embed-stop-error')).toHaveTextContent('crew unreachable')
    expect(screen.getByTestId('chat-embed-stop')).toBeInTheDocument()
  })
})
