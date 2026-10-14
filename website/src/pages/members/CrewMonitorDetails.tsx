import { useTranslation } from 'react-i18next'

import { intervalText } from '../../components/autoNudgeLoop'
import { fmtDateTime, fmtNumber } from '../../i18n/format'
import { isAutomationActive, type AutomationRecord } from '../../monitoring/automation'

export type MonitorReadState = 'loading' | 'ready' | 'error'

function translatedStopReason(reason: string, t: ReturnType<typeof useTranslation>['t']): string {
  switch (reason) {
    case 'manual': return t('pages.membersPage.profile_monitor_reason_manual')
    case 'autonudge_stop': return t('pages.membersPage.profile_monitor_reason_agent')
    case 'cycle_cap': return t('pages.membersPage.profile_monitor_reason_cycles')
    case 'runtime_budget': return t('pages.membersPage.profile_monitor_reason_runtime')
    case 'approval_stalled': return t('pages.membersPage.profile_monitor_reason_approval')
    case 'structural_terminal': return t('pages.membersPage.profile_monitor_reason_failed')
    case 'session_start_failures': return t('pages.membersPage.profile_monitor_reason_start')
    case 'monitor_terminal': return t('pages.membersPage.profile_monitor_reason_finished')
    default: return reason
  }
}

function stopReason(record: AutomationRecord | null): string {
  if (!record) return ''
  if (record.kind === 'legacy_goal_loop') return record.stoppedReason
  return record.terminal?.reason || (!record.actionable ? record.latest.reasonCode : '')
}

function monitorType(record: AutomationRecord | null, t: ReturnType<typeof useTranslation>['t']): string {
  if (!record) return '—'
  return record.kind === 'legacy_goal_loop'
    ? t('pages.membersPage.profile_monitor_type_loop')
    : t('pages.membersPage.profile_monitor_type_bounded')
}

function monitorSubject(record: AutomationRecord | null, t: ReturnType<typeof useTranslation>['t']): string {
  if (!record) return '—'
  if (record.kind === 'structured_monitor') return record.target || '—'
  if (record.monitorKind === 'work-ledger') return t('pages.membersPage.profile_monitor_subject_ledger')
  if (record.monitorKind === 'gh-pr') return t('pages.membersPage.profile_monitor_subject_pull_request')
  return record.monitorKind || t('pages.membersPage.profile_monitor_subject_goal')
}

function durationValue(seconds: number): string {
  return seconds > 0 ? intervalText(seconds) : '—'
}

export default function CrewMonitorDetails({
  automation,
  readState,
  nowTs,
}: {
  automation: AutomationRecord | null
  readState: MonitorReadState
  nowTs: number
}) {
  const { t } = useTranslation()
  const proven = readState === 'ready'
  const active = proven && isAutomationActive(automation)
  const unavailable = !proven || (automation?.kind === 'structured_monitor' && !automation.actionable)
  const status = unavailable
    ? t('pages.membersPage.profile_monitor_unavailable')
    : active
      ? t('pages.membersPage.profile_monitor_active')
      : t('pages.membersPage.profile_monitor_inactive')
  const cadence = automation
    ? durationValue(automation.kind === 'legacy_goal_loop' ? automation.idleSecs : automation.cadenceSecs)
    : '—'
  const currentCycle = automation
    ? fmtNumber(automation.kind === 'legacy_goal_loop' ? automation.cycleCount : automation.usage.probes)
    : '—'
  const maximumCycle = automation?.kind === 'legacy_goal_loop' && automation.maxCycles > 0
    ? fmtNumber(automation.maxCycles)
    : '—'
  const createdAt = automation?.createdAt ?? 0
  const nextAt = automation?.kind === 'legacy_goal_loop'
    ? automation.nextDueAt ?? 0
    : automation?.nextProbeAt ?? 0
  const maxRuntime = automation?.kind === 'legacy_goal_loop'
    ? automation.maxRuntimeSecs ?? 0
    : automation?.budgets.maxRuntimeSecs ?? 0
  const stoppedAt = automation?.kind === 'structured_monitor'
    ? automation.terminal?.stoppedAt ?? 0
    : 0
  const runtimeUsed = createdAt > 0 && (active || stoppedAt > 0)
    ? durationValue(Math.max(0, (active ? nowTs : stoppedAt) - createdAt))
    : '—'
  const runtime = maxRuntime > 0
    ? t('pages.membersPage.profile_monitor_runtime_value', {
        used: runtimeUsed,
        max: durationValue(maxRuntime),
      })
    : runtimeUsed
  const reason = stopReason(automation)
  const reasonText = reason ? translatedStopReason(reason, t) : '—'

  const rows = [
    [t('pages.membersPage.profile_monitor_state'), status],
    [t('pages.membersPage.profile_monitor_type'), monitorType(automation, t)],
    [t('pages.membersPage.profile_monitor_subject'), monitorSubject(automation, t)],
    [t('pages.membersPage.profile_monitor_frequency'), cadence],
    [t('pages.membersPage.profile_monitor_cycle'), t('pages.membersPage.profile_monitor_cycle_value', { current: currentCycle, max: maximumCycle })],
    [t('pages.membersPage.profile_monitor_started'), createdAt > 0 ? fmtDateTime(createdAt) : '—'],
    [t('pages.membersPage.profile_monitor_next'), active && nextAt > 0 ? fmtDateTime(nextAt) : '—'],
    [t('pages.membersPage.profile_monitor_runtime'), runtime],
    [t('pages.membersPage.profile_monitor_stopped'), active ? '—' : reasonText],
  ]

  return (
    <section
      className="rounded-2xl border border-border bg-bg px-3.5 py-3"
      data-testid="crew-profile-monitor"
      data-monitor-active={active ? 'true' : 'false'}
    >
      <h3 className="text-[11px] font-semibold tracking-wide uppercase text-muted mb-2">
        {t('pages.membersPage.profile_monitor')}
      </h3>
      <dl className="grid grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)] gap-x-3 gap-y-1.5 text-[12px] leading-snug">
        {rows.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-muted">{label}</dt>
            <dd className="min-w-0 break-words text-right font-medium text-text" title={value}>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
