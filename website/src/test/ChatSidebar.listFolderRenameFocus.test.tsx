/**
 * #13598 — ChatSidebar list-view folder rename: Enter and Escape restore focus
 * to the folder header row; a pointer blur moves nothing.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Provider } from 'react-redux'
import { MemoryRouter } from 'react-router-dom'
import { createTestStore } from './helpers'
import type { RootState } from '../store'
import { ThemeProvider } from '../hooks/useTheme'
import type { ChatFolder } from '../types'

vi.mock('framer-motion', async () => {
  const React = await import('react')
  const FRAMER_PROPS = new Set([
    'layout', 'layoutId', 'layoutScroll', 'initial', 'animate', 'exit',
    'transition', 'variants', 'whileHover', 'whileTap', 'whileInView',
    'drag', 'dragConstraints', 'dragElastic', 'onAnimationComplete',
  ])
  const make = (tag: string) =>
    React.forwardRef((props: Record<string, unknown>, ref: React.Ref<unknown>) => {
      const clean: Record<string, unknown> = {}
      for (const k of Object.keys(props)) {
        if (k === 'children') continue
        if (k === 'layoutId') { clean['data-layout-id'] = props[k]; continue }
        if (FRAMER_PROPS.has(k)) continue
        clean[k] = props[k]
      }
      return React.createElement(tag, { ...clean, ref }, props.children as React.ReactNode)
    })
  const motion = new Proxy({}, { get: (_t, tag: string) => make(tag) })
  return {
    motion,
    AnimatePresence: ({ children }: { children?: React.ReactNode }) => React.createElement(React.Fragment, null, children),
    LayoutGroup: ({ children }: { children?: React.ReactNode }) => React.createElement(React.Fragment, null, children),
  }
})

vi.mock('../components/ProjectPicker', () => ({ default: () => null }))
vi.mock('../pages/chat/ChatSettings', () => ({
  loadChatConfig: () => ({ tagColumnsEnabled: false, confirmCloseSession: false }),
  saveChatConfig: vi.fn(),
}))

const { updateChatFolderMock } = vi.hoisted(() => ({ updateChatFolderMock: vi.fn() }))

vi.mock('../api/client', () => ({
  SEARCH_MIN_CHARS: 2,
  api: new Proxy({} as Record<string, unknown>, {
    get: (_t, prop: string) => {
      if (prop === 'updateChatFolder') return updateChatFolderMock
      return vi.fn().mockResolvedValue([])
    },
  }),
}))

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((q: string) => ({
    matches: false, media: q, onchange: null,
    addListener: vi.fn(), removeListener: vi.fn(),
    addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn(),
  })),
})

import ChatSidebar from '../pages/ChatSidebar'

const FOLDER_ID = 'folder-focus-test'
const FOLDER_NAME = 'Focus Folder'

const folder: ChatFolder = {
  id: FOLDER_ID, name: FOLDER_NAME, collapsed: false, parent_id: '', color: '', icon: '',
}

function renderSidebar() {
  const store = createTestStore({
    dashboard: {
      status: {}, connected: true, slots: [], approvalMode: 'normal',
      channelTrusted: false, refreshTrigger: 0, unreadSlots: [], updateProgress: null,
      subagentRunning: {}, subagentDetails: {}, subagentText: {},
      sessionDefaultColor: null, sessionColorsMode: 'tint', sessionColorsPalette: 'horizon',
      sessionColorsIntensity: 'clear', slotsLoaded: true,
    } as unknown as RootState['dashboard'],
    chat: { activeSlot: null } as unknown as RootState['chat'],
  })
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  qc.setQueryData(['chat-tags'], [])
  qc.setQueryData(['tag-columns'], [])
  qc.setQueryData(['chat-folders'], [folder])
  const view = render(
    <QueryClientProvider client={qc}>
      <Provider store={store}>
        <ThemeProvider>
          <MemoryRouter>
            <ChatSidebar
              slots={[]} activeSlot={null} unreadSlots={[]}
              history={[]} historyHasMore={false} defaultAgent="" installedAgents={[]}
            />
            <button type="button" data-testid="elsewhere">Elsewhere</button>
          </MemoryRouter>
        </ThemeProvider>
      </Provider>
    </QueryClientProvider>,
  )
  return { store, container: view.container }
}

const nextFrame = () => act(async () => { await new Promise(requestAnimationFrame) })

/** Start a folder rename via double-click on the folder name. */
function startFolderRename(container: HTMLElement) {
  const folderRow = container.querySelector(`[data-folder-row="${FOLDER_ID}"]`) as HTMLElement
  expect(folderRow).toBeTruthy()
  const nameSpan = folderRow.querySelector('[title]') as HTMLElement
  expect(nameSpan).toBeTruthy()
  act(() => { fireEvent.doubleClick(nameSpan) })
  return container.querySelector('input[value]') as HTMLInputElement
}

beforeEach(() => {
  updateChatFolderMock.mockReset()
  updateChatFolderMock.mockResolvedValue({ ok: true })
})
afterEach(() => vi.clearAllMocks())

describe('ChatSidebar list folder rename — focus contract (#13598)', () => {
  it('folder-rename-status region is present and polite', async () => {
    const { container } = renderSidebar()
    await screen.findByText(FOLDER_NAME)
    const region = container.querySelector('[data-testid="folder-rename-status"]')
    expect(region).toBeTruthy()
    expect(region?.getAttribute('role')).toBe('status')
    expect(region?.getAttribute('aria-live')).toBe('polite')
    expect(region?.textContent).toBe('')
  })

  it('Enter commits and returns focus to the folder header', async () => {
    const { container } = renderSidebar()
    await screen.findByText(FOLDER_NAME)
    const input = startFolderRename(container)
    expect(input).toBeTruthy()
    act(() => { fireEvent.change(input, { target: { value: 'Renamed Folder' } }) })
    act(() => { fireEvent.keyDown(input, { key: 'Enter' }) })
    await nextFrame()
    await nextFrame()
    const folderRow = container.querySelector(`[data-folder-row="${FOLDER_ID}"]`) as HTMLElement
    const focusable = folderRow.querySelector<HTMLElement>('button,[tabindex]') ?? folderRow
    expect(document.activeElement).toBe(focusable)
    expect(document.activeElement).not.toBe(document.body)
  })

  it('Escape cancels and returns focus to the folder header', async () => {
    const { container } = renderSidebar()
    await screen.findByText(FOLDER_NAME)
    const input = startFolderRename(container)
    expect(input).toBeTruthy()
    act(() => { fireEvent.keyDown(input, { key: 'Escape' }) })
    await nextFrame()
    await nextFrame()
    const folderRow = container.querySelector(`[data-folder-row="${FOLDER_ID}"]`) as HTMLElement
    const focusable = folderRow.querySelector<HTMLElement>('button,[tabindex]') ?? folderRow
    expect(document.activeElement).toBe(focusable)
    expect(document.activeElement).not.toBe(document.body)
  })

  it('pointer blur does not redirect focus', async () => {
    const { container } = renderSidebar()
    await screen.findByText(FOLDER_NAME)
    const input = startFolderRename(container)
    expect(input).toBeTruthy()
    const elsewhere = screen.getByTestId('elsewhere')
    act(() => { elsewhere.focus() })
    act(() => { fireEvent.blur(input) })
    await nextFrame()
    await nextFrame()
    expect(document.activeElement).toBe(elsewhere)
    expect(document.activeElement).not.toBe(document.body)
  })
})
