import { beforeEach, describe, expect, it } from 'vitest'
import { claimDashboardOnLoad, resetDashboardOnLoad } from './dashboardOnLoad'

describe('claimDashboardOnLoad', () => {
  beforeEach(() => resetDashboardOnLoad())

  it('claims a crewmate once per page load', () => {
    expect(claimDashboardOnLoad('oncall', 'member-oncall')).toBe(true)
    // A later render or an in-app return keeps the tab the person picked.
    expect(claimDashboardOnLoad('oncall', 'member-oncall')).toBe(false)
  })

  it('claims again when the strip re-buckets onto the confirmed slot', () => {
    expect(claimDashboardOnLoad('oncall', '')).toBe(true)
    expect(claimDashboardOnLoad('oncall', 'member-oncall')).toBe(true)
  })

  it('claims each crewmate on its own', () => {
    expect(claimDashboardOnLoad('oncall', 's1')).toBe(true)
    expect(claimDashboardOnLoad('writer', 's2')).toBe(true)
  })

  it('claims again after a reload', () => {
    claimDashboardOnLoad('oncall', 's1')
    resetDashboardOnLoad()
    expect(claimDashboardOnLoad('oncall', 's1')).toBe(true)
  })
})
