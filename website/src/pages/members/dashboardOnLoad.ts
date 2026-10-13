/**
 * Which crewmates have already had their side panel put on the Dashboard tab in
 * THIS page load.
 *
 * The panel strip is persisted per slot (`usePanelTabs`), so a reload used to
 * reopen whatever tab was focused last -- Files, a terminal, a view. A refresh of
 * a crewmate opens its Dashboard tab instead: that tab is the crewmate's own
 * page, or the suggestions page when no page is adopted.
 *
 * Module-level on purpose: it empties on a reload and nowhere else, so moving
 * around the app and coming back keeps the tab the person picked. Keyed on the
 * crewmate AND the strip bucket (its confirmed slot), because the strip
 * re-buckets when the thread's slot confirms and that bucket's stored focus is
 * the one a reload restores.
 */
const opened = new Set<string>()

/** True the first time a crewmate's strip bucket is seen in this page load. */
export function claimDashboardOnLoad(member: string, slot: string): boolean {
  const key = JSON.stringify([member, slot])
  if (opened.has(key)) return false
  opened.add(key)
  return true
}

/** Test seam: a fresh page load. */
export function resetDashboardOnLoad(): void {
  opened.clear()
}
