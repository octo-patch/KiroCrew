import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import vm from 'node:vm'
import { beforeEach, describe, expect, it } from 'vitest'

type Item = Record<string, unknown>
type DashboardApi = {
  fields: Record<string, unknown>
  agentic: string[]
  written_at: Record<string, string>
  locale: string
  seq: number
  stale: boolean
}

type DashboardWindow = Window & { kirocrew: DashboardApi }

const root = resolve(__dirname, '../../..')
const html = readFileSync(resolve(root, 'src/kiro_crew/dashboard_templates/builtin/quest-log/template.html'), 'utf8')
const sample = JSON.parse(readFileSync(resolve(root, 'test/fixtures/dashboard_templates/sample_quest-log.json'), 'utf8')) as {
  agentic: { advice: unknown[] }
  agentic_written_at: Record<string, string>
  folds: { work: { seq: number; value: { conductor: Record<string, unknown>; items: Item[]; omitted: number; forgotten: Record<string, number> } } }
}
const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1]
const markup = html.replace(/<script>[\s\S]*?<\/script>/, '')
if (!script) throw new Error('quest-log template has no script')

function sampleFields(): Record<string, unknown> {
  const work = structuredClone(sample.folds.work.value)
  return {
    goal: work.conductor.goal,
    round: work.conductor.round,
    last_entry_at: work.conductor.last_entry_at,
    items: work.items,
    omitted: work.omitted,
    forgotten: work.forgotten,
    advice: structuredClone(sample.agentic.advice),
  }
}

function bind(fields: Record<string, unknown>): void {
  document.querySelectorAll<HTMLElement>('[data-dashboard-field]').forEach((node) => {
    const name = node.dataset.dashboardField ?? ''
    const value = fields[name]
    node.textContent = value === undefined || value === null
      ? ''
      : typeof value === 'object' ? JSON.stringify(value) : String(value)
  })
}

function render(fields = sampleFields(), writtenAt = sample.agentic_written_at, agentic = ['advice']): void {
  document.body.innerHTML = markup
  bind(fields)
  ;(window as DashboardWindow).kirocrew = {
    fields,
    agentic,
    written_at: writtenAt,
    locale: 'en',
    seq: sample.folds.work.seq,
    stale: false,
  }
  vm.runInNewContext(script, {
    window,
    document,
    parent: window.parent,
    setTimeout,
    Date,
    Object,
    Array,
    String,
    Math,
    RegExp,
    isFinite,
  })
}

beforeEach(() => {
  document.body.innerHTML = ''
})

describe('quest-log dashboard', () => {
  it('draws every section from its declared source', () => {
    render()

    expect(document.querySelector('.ql-goal')?.textContent).toBe('Ship the new right-panel task view')
    expect(document.querySelector('#ql-state')?.textContent).toBe('2 active · 1 needs you · 1 upcoming · 7 finished')
    expect(document.querySelectorAll('#ql-needs .ql-row')).toHaveLength(1)
    expect(document.querySelector('#ql-needs')?.textContent).toContain('Clear the blocker named below.')
    expect(document.querySelectorAll('#ql-active .ql-row')).toHaveLength(2)
    expect(document.querySelector('#ql-advice')?.textContent).toContain('Review the release choice before more work starts.')
    expect(document.querySelectorAll('#ql-upcoming .ql-row')).toHaveLength(1)
    expect(document.querySelector('#ql-upcoming')?.textContent).toContain('Write the rollout note')
    expect(document.querySelector('#ql-finished')?.textContent).toBe('7')
    expect(document.querySelector('#ql-omitted')?.textContent).toBe('3 log entries omitted')
  })

  it('keeps every section useful when no data or advice exists', () => {
    render({
      goal: '',
      round: 0,
      last_entry_at: '',
      items: [],
      omitted: 0,
      forgotten: {},
      advice: [],
    }, {}, [])

    expect(document.querySelector('.ql-goal')?.textContent).toBe('No current goal in the work log.')
    expect(document.querySelector('#ql-state')?.textContent).toBe('0 active · 0 needs you · 0 upcoming · 0 finished')
    expect(document.querySelector('#ql-needs')?.textContent).toContain('Nothing needs you right now.')
    expect(document.querySelector('#ql-active')?.textContent).toContain('No quest is active right now.')
    expect(document.querySelector('#ql-advice')?.textContent).toContain('No current advice.')
    expect(document.querySelector('#ql-upcoming')?.textContent).toContain('No queued quest or dependency is recorded.')
    expect(document.querySelectorAll('a')).toHaveLength(0)
  })

  it('hides advice older than the work fold', () => {
    render(sampleFields(), { advice: '2026-10-14T06:20:00Z' })

    expect(document.querySelector('#ql-advice')?.textContent).toContain('Advice is older than the work log')
    expect(document.querySelector('#ql-advice')?.textContent).not.toContain('Review the release choice')
  })

  it('renders hostile task and advice text as text only', () => {
    const fields = sampleFields()
    const items = fields.items as Item[]
    items[0] = {
      ...items[0],
      title: '<img src=x onerror="window.pwned=1">Long task',
      summary: '<script>window.pwned=2</script>',
    }
    fields.advice = [{ text: '<b>Do not trust markup</b>', why: '<svg onload="window.pwned=3"></svg>' }]
    render(fields)

    expect(document.querySelector('.ql img')).toBeNull()
    expect(document.querySelector('.ql svg')).toBeNull()
    expect(document.querySelector('.ql script')).toBeNull()
    expect(document.querySelector('#ql-needs')?.textContent).toContain('<img src=x onerror=')
    expect(document.querySelector('#ql-advice')?.textContent).toContain('<b>Do not trust markup</b>')
  })

  it('caps a board with more than fifty items and names the rest', () => {
    const fields = sampleFields()
    fields.items = Array.from({ length: 55 }, (_, index) => ({
      item_id: `it_${index}`,
      title: index === 0 ? 'A'.repeat(400) : `Task ${index}`,
      state: 'open',
      status: 'progress',
      summary: '',
      acceptance: { kind: 'file', path: `task-${index}` },
      pr: null,
    }))
    fields.forgotten = {}
    fields.omitted = 0
    fields.advice = []
    render(fields, {}, [])

    expect(document.querySelector('#ql-active-count')?.textContent).toBe('55')
    expect(document.querySelectorAll('#ql-active .ql-row')).toHaveLength(12)
    expect(document.querySelector('#ql-active .ql-more')?.textContent).toBe('43 more not shown')
    expect(document.querySelector('#ql-active .ql-title')?.textContent).toBe('A'.repeat(400))
  })

  it('makes each PR a safe real link', () => {
    render()

    const links = [...document.querySelectorAll<HTMLAnchorElement>('.ql-link')]
    expect(links).toHaveLength(2)
    for (const link of links) {
      expect(link.href).toMatch(/^https:\/\/github\.com\/kirodotdev\/KiroCrew\/pull\/2010[12]$/)
      expect(link.target).toBe('_blank')
      expect(link.rel).toBe('noopener noreferrer')
    }
  })
})
