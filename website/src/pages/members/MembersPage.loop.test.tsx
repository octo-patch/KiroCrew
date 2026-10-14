import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, within, fireEvent } from '@testing-library/react'
import { renderWithProviders } from '../../test/helpers'

/* Same api mock shape as MembersPage.filters.test.tsx. */
vi.mock('../../api/client', () => ({
  api: {
    members: vi.fn(),
    teams: { list: vi.fn(() => Promise.resolve({ teams: [] })) },
    memberThread: vi.fn((slug: string) =>
      Promise.resolve({ slot_key: 'member-' + slug, slug, member: slug, created: true }),
    ),
    memberActivity: vi.fn(() => Promise.resolve({ slug: '', member: '', capped: false, entries: [] })),
    crons: vi.fn(() => Promise.resolve({ jobs: [] })),
    webhooks: vi.fn(() => Promise.resolve({ tokens: [] })),
    defaultAgent: vi.fn(() => Promise.resolve({ default_agent: '' })),
    autonudgeList: vi.fn(() => Promise.resolve({ enabled: true, loops: [] })),
    monitorsList: vi.fn(() => Promise.resolve({ enabled: true, monitors: [] })),
  },
}))

vi.mock('../../components/ChatPane', () => ({
  default: ({ slotKey }: { slotKey: string }) => <div data-testid="chat-pane-stub">{slotKey}</div>,
}))

import { api } from '../../api/client'
import MembersPage from './MembersPage'
import { STRUCTURED_AUTOMATIONS_QUERY_KEY } from '../../monitoring/automation'
import { structuredMonitorLoop } from '../../test/monitorFixtures'

const LABEL = 'On watch'

function row(name: string) {
  return {
    name, slug: name, slot_key: `member-${name}`, running: false, kiro_agent: name,
    workspace: 'default', memory_store: 'default', model: '', source: 'kirocrew', starred: false,
  }
}

const monitor = (slotKey: string, patch: Record<string, unknown> = {}) => ({
  ...structuredMonitorLoop(patch),
  id: `monitor-${slotKey}`,
  slot_key: slotKey,
})

async function renderPage(route = '/members') {
  ;(api.members as ReturnType<typeof vi.fn>).mockResolvedValue({ members: [row('alpha'), row('beta')] })
  const utils = renderWithProviders(<MembersPage />, { route })
  await within(await screen.findByTestId('member-roster')).findByText('alpha')
  return utils
}

const rosterIndicators = () =>
  Array.from(document.querySelectorAll('[data-testid="member-loop-indicator"]'))

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  // clearAllMocks keeps implementations: reset the one a case overrides.
  vi.mocked(api.autonudgeList).mockResolvedValue({ enabled: true, loops: [] } as never)
  vi.mocked(api.monitorsList).mockResolvedValue({ enabled: true, monitors: [] } as never)
})

describe('MembersPage — on-watch loop indicator', () => {
  it('a live structured monitor marks every avatar surface, then stops at terminal state', async () => {
    vi.mocked(api.monitorsList).mockResolvedValue({
      enabled: true,
      monitors: [monitor('member-beta')],
    } as never)
    const { queryClient } = await renderPage('/members?member=beta')
    await waitFor(() => expect(rosterIndicators()).toHaveLength(1))
    const rosterMark = rosterIndicators()[0]
    expect(rosterMark.closest('li')?.textContent).toContain('beta')
    expect(rosterMark.getAttribute('aria-hidden')).toBe('true')
    expect(rosterMark.getAttribute('title')).toBe(LABEL)
    expect(rosterMark.closest('li')?.querySelector('[data-testid="member-loop-label"]')?.textContent).toBe(LABEL)
    const pill = await screen.findByTestId('member-identity-pill')
    expect(within(pill).getByTestId('member-pill-loop-indicator')).toBeInTheDocument()

    vi.mocked(api.monitorsList).mockResolvedValue({
      enabled: true,
      monitors: [monitor('member-beta', {
        active: false,
        outcome: 'success',
        stopped_reason: 'monitor_terminal',
        stopped_at: 1,
      })],
    } as never)
    await queryClient.invalidateQueries({ queryKey: STRUCTURED_AUTOMATIONS_QUERY_KEY })
    await waitFor(() => expect(rosterIndicators()).toHaveLength(0))
    expect(within(pill).queryByTestId('member-pill-loop-indicator')).toBeNull()
  })

  it('an active legacy goal loop shows the same label, never its cycle count', async () => {
    vi.mocked(api.autonudgeList).mockResolvedValue({
      enabled: true,
      loops: [{ id: 'l1', slot_key: 'member-alpha', active: true, cycle_count: 3, max_cycles: 24, message: 'nudge text' }],
    } as never)
    await renderPage()
    await waitFor(() => expect(rosterIndicators()).toHaveLength(1))
    expect(rosterIndicators()[0].getAttribute('title')).toBe(LABEL)
    expect(document.body.textContent).not.toMatch(/3 of 24|On patrol/)
  })

  it('fails closed when either monitor registry cannot be read', async () => {
    vi.mocked(api.autonudgeList).mockResolvedValue({
      enabled: true,
      loops: [{ id: 'l1', slot_key: 'member-alpha', active: true, cycle_count: 3, max_cycles: 24, message: 'nudge text' }],
    } as never)
    vi.mocked(api.monitorsList).mockRejectedValue(new Error('owner read failed'))
    await renderPage('/members?member=alpha')
    await waitFor(() => expect(screen.getByTestId('member-main-patrol-error')).toBeInTheDocument())
    expect(rosterIndicators()).toHaveLength(0)
    expect(screen.queryByTestId('member-pill-loop-indicator')).toBeNull()
  })

  it('the Profile avatar and details use the same active record', async () => {
    vi.mocked(api.monitorsList).mockResolvedValue({
      enabled: true,
      monitors: [monitor('member-alpha', { created_ts: 1_799_996_400 })],
    } as never)
    await renderPage('/members?member=alpha')
    const pill = await screen.findByTestId('member-identity-pill')
    const pillMark = await within(pill).findByTestId('member-pill-loop-indicator')
    expect(pillMark.getAttribute('title')).toBe(LABEL)
    expect(screen.getByTestId('member-pill-activity-sr').textContent).toContain(LABEL)

    fireEvent.click(pill)
    const card = await screen.findByTestId('crew-profile-panel')
    expect(within(card).getByTestId('crew-profile-loop-indicator')).toBeInTheDocument()
    fireEvent.click(within(card).getByRole('tab', { name: 'Profile' }))
    const details = within(card).getByTestId('crew-profile-monitor')
    expect(details).toHaveAttribute('data-monitor-active', 'true')
    expect(details).toHaveTextContent('https://github.com/owner/repo/pull/123')
    expect(details).toHaveTextContent('5m')
    expect(details).toHaveTextContent('5 / —')
  })
})
