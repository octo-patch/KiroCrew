import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import type { ReactNode } from 'react'
import { act, render, screen, fireEvent, waitFor } from '@testing-library/react'
import type { RootState } from '../store'
import { store as appStore } from '../store'
import { Provider } from 'react-redux'
import { MemoryRouter } from 'react-router-dom'
import { configureStore } from '@reduxjs/toolkit'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from '../hooks/useTheme'
import chatReducer, { appendSlotMessage, selectSlotMessages, setActiveSlot, setQuestionCard, sseChatMessage, sseSubagentPending } from '../store/chatSlice'
import dashboardReducer from '../store/dashboardSlice'
import notificationsReducer from '../store/notificationsSlice'
import { __resetPaneDraftsForTests } from '../utils/chatPaneDrafts'
import type { ChatMessage } from '../types'
import * as transport from '../chat-core/transport/sendTurn'

// The real pane/composer/serializer/reducers; only network and unrelated rich
// content are mocked. A deferred receipt lets each test observe the send itself.
vi.mock('react-virtuoso', () => ({
  Virtuoso: ({ data, itemContent }: { data?: unknown[]; itemContent: (index: number, item: unknown) => ReactNode }) => (
    <div data-testid="virtuoso">{data?.map((d: unknown, i: number) => <div key={i}>{itemContent(i, d)}</div>)}</div>
  ),
}))
vi.mock('../api/client', () => ({
  api: {
    chatSlots: vi.fn().mockResolvedValue([]),
    chatSlotDetail: vi.fn(),
    sendChat: vi.fn(),
    chatHistory: vi.fn().mockResolvedValue({ sessions: [] }),
    models: vi.fn().mockResolvedValue([]),
    agents: vi.fn().mockResolvedValue([]),
    agentDetail: vi.fn().mockResolvedValue({}),
    workspaces: vi.fn().mockResolvedValue({ workspaces: [] }),
    spawnList: vi.fn().mockResolvedValue({ agents: [] }),
    uploadFiles: vi.fn(),
    screenshot: vi.fn().mockResolvedValue({ path: null }),
    fileSearch: vi.fn().mockResolvedValue({ root: '/repo', results: [] }),
    chatSlotAgent: vi.fn().mockResolvedValue(undefined),
    stopChat: vi.fn(),
    interruptSlot: vi.fn(),
    resolveApproval: vi.fn().mockResolvedValue({ ok: true }),
    answerQuestion: vi.fn().mockResolvedValue({ ok: true }),
  },
  SEARCH_MIN_CHARS: 2,
  ApiError: class ApiError extends Error {},
}))
vi.mock('../hooks/useVoiceInput', () => ({ useVoiceInput: () => ({ recording: false, transcribing: false, toggle: vi.fn() }), voiceInputSupported: false }))
vi.mock('../hooks/useBranding', () => ({ useBranding: () => ({ botName: 'Test', avatar: '' }) }))
vi.mock('../hooks/useAgents', () => ({ useAgents: () => ({ agents: [{ name: 'default' }], defaultAgent: 'default' }) }))
vi.mock('../components/MarkdownRenderer', () => ({ default: ({ content }: { content: string }) => <span>{content}</span> }))
vi.mock('../hooks/useWebSocket', () => ({ useWebSocket: () => ({ subscribeLogs: () => {} }) }))
vi.mock('../pages/chat/SubagentProgressBar', () => ({ default: () => <div data-testid="subagent-progress-probe" /> }))

import ChatPane from '../components/ChatPane'
import CrewComposer from '../pages/members/CrewComposer'
import { api } from '../api/client'

const SLOT = 'member-radar'
const FIXED_TIME = new Date('2026-10-14T07:00:00Z')
let settleSend: (response: Response) => void
const response = (body: object, status = 200) => ({ ok: status < 400, status, json: async () => body }) as Response

function renderCrewmate(opts: { running?: boolean; delegated?: boolean; ordinary?: boolean; stopping?: boolean; messages?: ChatMessage[]; width?: number } = {}) {
  const { running = true, delegated = false, ordinary = false, stopping = false, messages = [], width = 1024 } = opts
  vi.stubGlobal('innerWidth', width)
  vi.stubGlobal('matchMedia', vi.fn((query: string) => ({
    matches: query.includes('max-width') && width < 768,
    media: query, onchange: null, addEventListener: vi.fn(), removeEventListener: vi.fn(),
    addListener: vi.fn(), removeListener: vi.fn(), dispatchEvent: vi.fn(),
  })))
  vi.mocked(api.chatSlotDetail).mockResolvedValue({ messages, running, has_more: false, total: messages.length } as never)
  const store = configureStore({
    reducer: { dashboard: dashboardReducer, chat: chatReducer, notifications: notificationsReducer },
    preloadedState: {
      dashboard: {
        status: null, connected: true,
        slots: [{ key: SLOT, messages: messages.length, running, stopping, stop_state: stopping ? 'soft_pending' : 'idle', subagents_running: delegated, mode: 'member', pending_approval: false, waiting_for_input: false }],
        slotsLoaded: true, unreadSlots: [], refreshTrigger: 0, approvalMode: 'normal',
        subagentRunning: {}, subagentDetails: {}, subagentText: {},
      } as unknown as RootState['dashboard'],
    } as Partial<RootState>,
  })
  store.dispatch(setActiveSlot('front'))
  vi.spyOn(appStore, 'getState').mockImplementation(store.getState)
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const view = render(
    <Provider store={store}><QueryClientProvider client={qc}><ThemeProvider><MemoryRouter>
      <ChatPane slotKey={SLOT} crewmate={ordinary ? undefined : { name: 'Radar' }} composerInput={ordinary ? undefined : CrewComposer} />
    </MemoryRouter></ThemeProvider></QueryClientProvider></Provider>,
  )
  return { ...view, store }
}

const userRows = (store: ReturnType<typeof renderCrewmate>['store']) =>
  selectSlotMessages(store.getState() as RootState, SLOT).filter(m => m.role === 'user')

beforeEach(() => {
  vi.clearAllMocks()
  vi.setSystemTime(FIXED_TIME)
  __resetPaneDraftsForTests()
  vi.mocked(api.chatSlotDetail).mockReset()
  vi.mocked(api.uploadFiles).mockReset().mockResolvedValue({ paths: ['/uploads/report.pdf'] } as never)
  vi.mocked(api.sendChat).mockReset().mockImplementation(() => new Promise<Response>(resolve => { settleSend = resolve }))
})
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers() })

async function ready() {
  const box = await screen.findByPlaceholderText(/Message (Radar|Test)/)
  await waitFor(() => expect(api.chatSlotDetail).toHaveBeenCalled())
  return box
}
async function send(text: string, click = false) {
  const box = await ready()
  fireEvent.change(box, { target: { value: text } })
  if (click) fireEvent.click(screen.getByRole('button', { name: 'Send', exact: true }))
  else fireEvent.keyDown(box, { key: 'Enter', code: 'Enter' })
  await waitFor(() => expect(api.sendChat).toHaveBeenCalledTimes(1))
  return box
}
async function settle(body: object = { ok: true }, status = 200) {
  await act(async () => { settleSend(response(body, status)) })
}

describe('crewmate responsive Send', () => {
  it.each([
    { running: false, width: 320 },
    { running: true, width: 390 },
    { running: false, delegated: true, width: 320 },
    { running: true, stopping: true, width: 390 },
  ])('keeps an ordinary Send while $running/$delegated/$stopping at $width px', async opts => {
    const { store, container } = renderCrewmate(opts)
    await ready()
    const button = screen.getByRole('button', { name: 'Send', exact: true })
    expect(button).toBeDisabled()
    expect(button.querySelector('.lucide-arrow-up')).not.toBeNull()
    expect(screen.queryByTestId('busy-send-split')).toBeNull()
    expect(screen.queryByRole('button', { name: /stop|interrupt/i })).toBeNull()
    const box = await send('also check the logs', true)
    expect(userRows(store)).toHaveLength(1)
    expect(userRows(store)[0].meta).toMatchObject({ optimistic: true, responsiveSend: true })
    expect(screen.getAllByText('also check the logs')).toHaveLength(1)
    expect(box).toHaveValue('')
    expect(vi.mocked(api.sendChat).mock.calls[0][5]).toBeUndefined()
    expect(vi.mocked(api.sendChat).mock.calls[0][6]).toBe(true)
    expect(api.stopChat).not.toHaveBeenCalled()
    expect(api.interruptSlot).not.toHaveBeenCalled()
    // DOM contract only: happy-dom cannot measure pixel overflow.
    expect(container.querySelector('[data-testid="composer-sizer"]')).not.toBeNull()
    expect(screen.getByTestId('composer-dock-root')).toHaveClass('absolute', 'left-0', 'bottom-0')
    await settle()
    expect(userRows(store)[0].meta?.optimistic).toBeUndefined()
  })

  it.each(['ordinary', 'steer'])('the %s echo settles the same bubble, even before a lost receipt', async echo => {
    const { store } = renderCrewmate()
    const box = await send('check once')
    const sendId = vi.mocked(api.sendChat).mock.calls[0][4]!.sendId
    act(() => {
      const meta = { sendId, mid: 'mid-fixed', ...(echo === 'steer' ? { steer: true, steerState: 'consumed' } : {}) }
      if (echo === 'steer') store.dispatch(appendSlotMessage({ slot: SLOT, message: { role: 'user', content: 'check once', cls: 'msg msg-u', ts: '2026-10-14T07:00:01Z', meta } }))
      else store.dispatch(sseChatMessage({ slot: SLOT, role: 'user', content: 'check once', ts: '2026-10-14T07:00:01Z', meta }))
    })
    expect(userRows(store)).toHaveLength(1)
    expect(userRows(store)[0].meta?.optimistic).toBeUndefined()
    expect(screen.queryByText(/Steered|Steering|runs as its own/)).toBeNull()
    await settle({ ok: true, steered: echo === 'steer' })
    expect(box).toHaveValue('')
    expect(userRows(store)).toHaveLength(1)
  })

  it('separates streaming text around the send without sending an interrupt', async () => {
    const { store } = renderCrewmate()
    await ready()
    act(() => { store.dispatch(sseChatMessage({ slot: SLOT, role: 'chunk', content: 'Before', seq: 1 })) })
    await send('new request')
    act(() => { store.dispatch(sseChatMessage({ slot: SLOT, role: 'chunk', content: 'After', seq: 2 })) })
    expect(selectSlotMessages(store.getState() as RootState, SLOT).map(m => [m.role, m.content])).toEqual([
      ['assistant', 'Before'], ['user', 'new request'], ['streaming', 'After'],
    ])
    await settle()
  })

  it('restores attachments and the typed draft on a definite refusal', async () => {
    const { store, container } = renderCrewmate()
    const box = await ready()
    const upload = container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(upload, { target: { files: [new File(['report'], 'report.pdf', { type: 'application/pdf' })] } })
    await waitFor(() => expect(api.uploadFiles).toHaveBeenCalled())
    await screen.findByText('report.pdf')
    await send('read this')
    expect(vi.mocked(api.sendChat).mock.calls[0][0]).toContain('[attached_file 1] /uploads/report.pdf')
    expect(vi.mocked(api.sendChat).mock.calls[0][4]).toMatchObject({ files: ['/uploads/report.pdf'] })
    await settle({ ok: false, error: 'request refused' }, 409)
    await waitFor(() => expect(box).toHaveValue('read this'))
    expect(userRows(store)).toHaveLength(0)
    expect(screen.getByText('report.pdf')).toBeInTheDocument()
    expect(screen.getByText(/request refused/)).toBeInTheDocument()
  })

  it('a real queued receipt replaces the provisional bubble without offering a client interrupt', async () => {
    const { store } = renderCrewmate()
    await send('queued safely')
    act(() => {
      store.dispatch(appendSlotMessage({ slot: SLOT, message: { role: 'queued', content: 'queued safely', cls: '', meta: { queueId: 'queue-fixed' } } }))
    })
    await settle({ ok: true, queued: true, queue_id: 'queue-fixed' })
    expect(userRows(store)).toHaveLength(0)
    expect(screen.getAllByText('queued safely')).toHaveLength(1)
    expect(screen.getByTestId('queue-card')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /interrupt/i })).toBeNull()
  })

  it('an uncertain receipt keeps its bubble with delivery-pending text, without restoring a duplicate', async () => {
    const { store } = renderCrewmate()
    await ready()
    vi.spyOn(transport, 'sendTurn').mockResolvedValue({ status: 'response-late', body: {} })
    const box = await screen.findByPlaceholderText(/Message (Radar|Test)/)
    fireEvent.change(box, { target: { value: 'keep pending' } })
    fireEvent.keyDown(box, { key: 'Enter', code: 'Enter' })
    await waitFor(() => expect(userRows(store)[0]?.meta?.deliveryUnconfirmed).toBe(true))
    expect(box).toHaveValue('')
    expect(screen.getByText(/Delivery pending/)).toBeInTheDocument()
  })

  it('keeps pending questions while hiding engineering rows and historical steer badges', async () => {
    const messages: ChatMessage[] = [
      { role: 'user', content: 'earlier request', cls: '', meta: { steer: true, steerState: 'requeued' } },
      { role: 'assistant', content: 'visible reply', cls: '' },
      { role: 'notice', content: 'runtime notice', cls: '' },
      { role: 'tool', content: '🔧 hidden shell command', cls: '' },
      { role: 'subagent', content: '[Subagent completion event]\nWorker complete', cls: '' },
      { role: 'system', content: 'hidden stop detail', cls: '', kind: 'stop_event', meta: { kind: 'stop_event' } },
      { role: 'error', content: 'actionable failure', cls: '' },
      { role: 'permission', content: 'approve required action', cls: '', meta: { approval_id: 'approval-fixed', tool: 'shell', description: 'approve required action' } },
    ]
    const { store } = renderCrewmate({ messages })
    await screen.findByText('visible reply')
    act(() => { store.dispatch(setQuestionCard({ slot: SLOT, ask_id: 'ask-fixed', questions: [{ question: 'Which report?', options: [{ label: 'Summary' }, { label: 'Full' }] }] })) })
    expect(await screen.findByText('Which report?')).toBeInTheDocument()
    expect(screen.getByText(/actionable failure/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Allow once' })).toBeInTheDocument()
    await send('also check this')
    expect(screen.getByRole('button', { name: 'Allow once' })).toBeInTheDocument()
    await settle()
    expect(screen.queryByText(/runtime notice|hidden shell|hidden stop|Worker complete|runs as its own/)).toBeNull()
    expect(screen.queryByTestId('subagent-progress-probe')).toBeNull()
    expect(screen.getByText('earlier request')).toBeInTheDocument()
    expect(selectSlotMessages(store.getState() as RootState, SLOT).some(m => m.role === 'permission' && !m.meta?.resolved)).toBe(true)
    expect(selectSlotMessages(store.getState() as RootState, SLOT).some(m => m.kind === 'stop_event')).toBe(true)
  })

  it('keeps IME and Shift+Enter local, then sends normally on a busy DM', async () => {
    renderCrewmate({ width: 320 })
    const box = await ready()
    fireEvent.change(box, { target: { value: '你好' } })
    fireEvent.compositionStart(box)
    fireEvent.keyDown(box, { key: 'Enter', code: 'Enter', isComposing: true })
    expect(api.sendChat).not.toHaveBeenCalled()
    fireEvent.compositionEnd(box)
    fireEvent.keyDown(box, { key: 'Enter', code: 'Enter', shiftKey: true })
    expect(api.sendChat).not.toHaveBeenCalled()
    // A fresh keydown after the composition commit uses the normal Send path.
    fireEvent.keyUp(box, { key: 'Enter', code: 'Enter' })
    fireEvent.click(screen.getByRole('button', { name: 'Send', exact: true }))
    await waitFor(() => expect(api.sendChat).toHaveBeenCalledTimes(1))
    expect(vi.mocked(api.sendChat).mock.calls[0][6]).toBe(true)
    await settle()
  })

  it('retains actionable spawn approvals even though routine subagent progress is hidden', async () => {
    const { store } = renderCrewmate({ width: 320 })
    await ready()
    act(() => { store.dispatch(sseSubagentPending({ slot: SLOT, id: 'worker-fixed', task: 'Review report', approval_id: 'spawn-fixed' })) })
    const card = await screen.findByTestId('spawn-approval-card')
    expect(card).toBeInTheDocument()
    expect(screen.queryByTestId('subagent-progress-probe')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Approve', exact: true }))
    await waitFor(() => expect(api.resolveApproval).toHaveBeenCalledWith('spawn-fixed', 'approve'))
  })

  it('an ordinary Sessions pane retains its engineering affordances and sends no responsive flag', async () => {
    const { store } = renderCrewmate({ ordinary: true, messages: [{ role: 'notice', content: 'runtime notice', cls: '' }] })
    await screen.findByText('runtime notice')
    act(() => { store.dispatch(sseChatMessage({ slot: SLOT, role: 'chunk', content: 'working', seq: 1 })) })
    expect(screen.getByTestId('subagent-progress-probe')).toBeInTheDocument()
    await send('normal steer')
    expect(vi.mocked(api.sendChat).mock.calls[0][5]).toBe(true)
    expect(vi.mocked(api.sendChat).mock.calls[0]).toHaveLength(6)
    await settle({ ok: true, steered: true })
  })
})


describe('responsive question and queue repairs', () => {
  it.each([
    { running: true, native: true, steer: true },
    { running: false, native: true, steer: false },
    { running: false, native: false, steer: false },
  ])('answers a card through the correct route: %j', async ({ running, native, steer }) => {
    const { store } = renderCrewmate({ running })
    await ready()
    act(() => { store.dispatch(setQuestionCard({ slot: SLOT, card_id: 'native-card', native, questions: [{ question: 'Which?', options: [{ label: 'First' }] }] })) })
    fireEvent.click(await screen.findByText('First'))
    fireEvent.click(screen.getByText('Submit'))
    await waitFor(() => expect(api.sendChat).toHaveBeenCalledTimes(1))
    const args = vi.mocked(api.sendChat).mock.calls[0]
    expect(!!args[5]).toBe(steer)
    expect(args[6]).toBe(true)
    await settle(steer ? { ok: true, steered: true } : { ok: true })
    expect(userRows(store)).toHaveLength(1)
  })

  it('answers a blocking MCP card through its answer endpoint, not a chat turn', async () => {
    const { store } = renderCrewmate()
    await ready()
    act(() => { store.dispatch(setQuestionCard({ slot: SLOT, ask_id: 'blocking-ask', questions: [{ question: 'Which?', options: [{ label: 'First' }] }] })) })
    fireEvent.click(await screen.findByText('First'))
    fireEvent.click(screen.getByText('Submit'))
    await waitFor(() => expect(api.answerQuestion).toHaveBeenCalledWith('blocking-ask', { 'Which?': 'First' }))
    expect(api.sendChat).not.toHaveBeenCalled()
  })

  it.each([
    { drained: false, status: 'response-late' as const },
    { drained: true, status: 'response-late' as const },
    { drained: false, status: 'refused' as const },
    { drained: true, status: 'refused' as const },
    { drained: false, status: 'transport-error' as const },
    { drained: true, status: 'transport-error' as const },
  ])('queue identity reconciles before a receipt: %j', async ({ drained, status }) => {
    const { appendQueuedMessage, removeQueuedMessage } = await import('../store/chatSlice')
    const { store } = renderCrewmate()
    await ready()
    let finish!: (receipt: transport.SendReceipt) => void
    vi.spyOn(transport, 'sendTurn').mockImplementation(() => new Promise(resolve => { finish = resolve }))
    const box = screen.getByPlaceholderText(/Message (Radar|Test)/)
    fireEvent.change(box, { target: { value: 'exact queue message' } })
    fireEvent.keyDown(box, { key: 'Enter', code: 'Enter' })
    await waitFor(() => expect(transport.sendTurn).toHaveBeenCalledTimes(1))
    const sendId = userRows(store)[0].meta!.sendId as string
    act(() => {
      store.dispatch(appendQueuedMessage({ slot: SLOT, content: 'exact queue message', ts: FIXED_TIME.toISOString(), queue_id: 'q-exact', meta: { sendId } }))
      if (drained) store.dispatch(removeQueuedMessage({ slot: SLOT, content: 'exact queue message', queue_id: 'q-exact', meta: { sendId } }))
    })
    expect(selectSlotMessages(store.getState() as RootState, SLOT).filter(m => m.content === 'exact queue message')).toHaveLength(1)
    await act(async () => { finish({ status, body: {} }) })
    expect(screen.queryByText(/Delivery pending/)).toBeNull()
    expect(box).toHaveValue('')
    expect(screen.getAllByText('exact queue message')).toHaveLength(1)
    expect(!!screen.queryByTestId('queue-card')).toBe(!drained)
  })
})
