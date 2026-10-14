/**
 * The two update switches and the one rule each follows, shared by Settings →
 * About, the post-update "What's new" modal and the settings searches (the
 * Command Bar, the command palette and the Settings page search).
 *
 * - The APP switch (the desktop updater's auto-download opt-out) is offered
 *   wherever the window's bridge can read and write that preference and the
 *   updater is not disabled.
 * - The GATEWAY switch (`auto_update`) follows the gateway's own
 *   `update_auto_effect` and nothing about the desktop updater. In a browser
 *   window the gateway card IS the update UI, so the row is always drawn and
 *   says when the switch cannot install or when the gateway has not said yet.
 *   In a desktop window it is drawn when the effect is a definite install or
 *   mandatory, or when the status frame reports the connected gateway is not the
 *   bundled one (the app reuses an external gateway that manages its own bytes).
 */
import { useQuery } from '@tanstack/react-query'
import { useAppSelector } from '../store'
import { hasAutoDownload, updateInfoQuery } from '../api/updateInfoQuery'
import { i18nT } from '../i18n/t'
import type { StatusData } from '../types'

type Effect = NonNullable<StatusData['update_auto_effect']>

/**
 * The gateway's `update_auto_effect`, which its update loop branches on.
 * Absent (a gateway without the field) or unrecognised reads as `unknown`.
 */
export function gatewayAutoUpdateEffect(raw: unknown): Effect {
  return raw === 'install' || raw === 'notify' || raw === 'mandatory' ? raw : 'unknown'
}

/** The status frame's effect, read the one way every surface reads it. */
export function useGatewayAutoUpdateEffect(): Effect {
  return gatewayAutoUpdateEffect(useAppSelector(s => s.dashboard.status?.update_auto_effect))
}

/**
 * Which update switches a window draws (see the module rule). `info` is the
 * updater's `getInfo()` answer, `undefined` while it has not answered.
 * `bundledByApp` is `update_bundled_by_app` from the status frame: `true` when
 * this gateway is the one the desktop app bundled and launched, `false` when the
 * app reuses an external gateway (a CLI service, an SSH-forwarded one), and
 * `undefined` while the status has not arrived. A reused gateway in a desktop
 * window has its own update surface, so its switch is always drawn even when the
 * effect is not yet a definite install.
 */
export function updateSwitchesShown({ bridge, info, effect, bundledByApp }: {
  bridge: UpdateAPI | undefined
  info: { disabled?: string } | null | undefined
  effect: Effect
  bundledByApp?: boolean
}): { app: boolean; gateway: boolean } {
  return {
    app: hasAutoDownload(bridge) && !info?.disabled,
    // In a desktop window, show the gateway switch when the effect is a definite
    // install or mandatory (the existing rule), OR when the gateway is known to be
    // a reused one (not the bundled gateway): a reused gateway manages its own
    // updates and the app's updater does not cover it.
    gateway: !bridge || bundledByApp === false || effect === 'install' || effect === 'mandatory',
  }
}

/**
 * Which update switches THIS window draws: the rule above, fed from the
 * window's own bridge, the updater's `['update-info']` answer and the status
 * frame's effect, so every surface asks it the same way. `fetch: false` reads
 * the updater's answer from cache only (the Command Bar's root issues no
 * request).
 */
export function useUpdateSwitchesShown({ fetch = true }: { fetch?: boolean } = {}): { app: boolean; gateway: boolean } {
  const bridge = window.updateAPI
  const { data: info } = useQuery({ ...updateInfoQuery, enabled: fetch && !!bridge })
  const bundledByApp = useAppSelector(s => s.dashboard.status?.update_bundled_by_app)
  return updateSwitchesShown({ bridge, info, effect: useGatewayAutoUpdateEffect(), bundledByApp })
}

/**
 * How the gateway switch reads for an effect. The label and the description
 * (what the switch does where it can) are the same on every install; this is
 * the part that changes: whether the switch can be flipped, and the note under
 * it. `canApply` tells the two notify cases apart: an install whose Update
 * action works (a checkout off its primary branch) and one with nothing to
 * apply with (a foreign wheel, a policy provider with no apply command).
 */
export function gatewayAutoUpdateCopy(effect: Effect, { canApply }: { canApply: boolean }): {
  disabled: boolean
  hint?: string
} {
  switch (effect) {
    case 'notify':
      return {
        disabled: true,
        hint: canApply
          ? i18nT('pages.settings.aboutPanel.auto_update_notify_manual_apply')
          : i18nT('pages.settings.aboutPanel.auto_update_notify_only_on_this_install'),
      }
    case 'mandatory':
      return { disabled: false, hint: i18nT('pages.settings.aboutPanel.gateway_auto_update_mandatory_note') }
    case 'unknown':
      return { disabled: false, hint: i18nT('pages.settings.aboutPanel.gateway_auto_update_unknown_hint') }
    case 'install':
      return { disabled: false }
  }
}
