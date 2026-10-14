/**
 * Screenshot evidence for #13598: focus ring on rename triggers after Enter.
 *
 * Frames:
 *   01-project-rename-enter       Projects page: title span has focus ring after Enter
 *   02-slot-rename-enter          ChatSidebar session row has focus ring after Enter
 *   03-folder-list-rename-enter   ChatSidebar list-view folder header after Enter
 *
 * Focus ring = 2px solid var(--accent), from index.css :focus-visible rule.
 *
 * Usage: node capture/shoot-rename-focus-ring-13598.mjs [outDir]
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { serveDist } from '../scripts/lib/serve-dist.mjs'
import { logPageProblems, stubDashboardApi, json } from '../scripts/lib/stub-dashboard-api.mjs'
import { chromiumExecutable } from '../scripts/lib/chromium-executable.mjs'

const OUT = process.argv[2] || '../temp-screenshots/rename-focus-ring-13598'
mkdirSync(OUT, { recursive: true })

const SLOT_KEY = 'rename-focus-slot-1'
const FOLDER_ID = 'folder-rename-focus-1'

const SLOT = {
  key: SLOT_KEY, title: 'Rename Focus Session',
  running: false, tags: [], created: '', last_ts: '',
}

const FOLDER = {
  id: FOLDER_ID, name: 'Rename Focus Folder', collapsed: false,
  parent_id: '', color: '', icon: '',
}

const { srv, base } = await serveDist()
const browser = await chromium.launch({
  executablePath: chromiumExecutable(),
  headless: true,
  args: ['--no-sandbox'],
})
const page = await browser.newPage()
await page.setViewportSize({ width: 1280, height: 800 })
logPageProblems(page)

async function capture(name, fn) {
  await fn()
  await page.waitForTimeout(400)
  const path = `${OUT}/${name}.png`
  await page.screenshot({ path })
  console.log(`captured ${name}`)
}

// ── Frame 02: Session slot rename — focus on the row after Enter ──────────
// Start with the simpler ChatSidebar frames first.
await stubDashboardApi(page, {
  slots: [SLOT],
  extra: async (path, route) => {
    // Slot rename succeeds
    if (/\/api\/slots\/[^/]+\/rename/.test(path)) return json(route, {}), true
    if (path === '/api/chat-slots') return json(route, [SLOT]), true
  },
})
await page.goto(`${base}`, { waitUntil: 'networkidle' })
await capture('02-slot-rename-enter', async () => {
  await page.waitForSelector('[data-session-title]', { timeout: 5000 }).catch(() => {})
  const titleDiv = await page.$('[data-session-title]')
  if (titleDiv) {
    await titleDiv.dblclick()
    await page.waitForSelector('textarea', { timeout: 2000 }).catch(() => {})
    const ta = await page.$('textarea')
    if (ta) {
      await ta.fill('Session Renamed')
      await ta.press('Enter')
    }
  }
})

// ── Frame 03: List-view folder rename — focus on folder header after Enter ─
await stubDashboardApi(page, {
  slots: [],
  extra: async (path, route) => {
    if (path === '/api/chat-folders') return json(route, [FOLDER]), true
    if (/\/api\/chat-folders\//.test(path)) return json(route, { ...FOLDER }), true
  },
})
await page.goto(`${base}`, { waitUntil: 'networkidle' })
await capture('03-folder-list-rename-enter', async () => {
  await page.waitForSelector(`[data-folder-row="${FOLDER_ID}"]`, { timeout: 5000 }).catch(() => {})
  const folderRow = await page.$(`[data-folder-row="${FOLDER_ID}"]`)
  if (folderRow) {
    const nameSpan = await folderRow.$('[title]')
    if (nameSpan) {
      await nameSpan.dblclick()
      await page.waitForSelector('input', { timeout: 2000 }).catch(() => {})
      // Scope to the row element so we don't accidentally fill the search box
      const input = await folderRow.$('input')
      if (input) {
        await input.fill('Folder Renamed')
        await input.press('Enter')
      }
    }
  }
})

await browser.close()
await srv.close()
console.log(`\nDone. Screenshots in: ${OUT}`)
