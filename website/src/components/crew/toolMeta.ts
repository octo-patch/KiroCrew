/**
 * Front-end-only metadata for the built-in tools, used by the Built From → Tools
 * read-only inspector to turn bare tool IDs into something a person can read:
 * a friendly name, a one-line description, a category for grouping, and whether
 * the tool is read-only or can modify/run things.
 *
 * This is a DISPLAY aid, not a source of truth. The read-only set mirrors the
 * backend's `_HOST_READ_ONLY_BUILTIN_TOOLS` (hooks.py) at the time of writing;
 * it is only used to tint a badge, never to make a security decision. An unknown
 * tool id (e.g. an MCP tool) falls back to its id, the "Other" group, and an
 * unknown mutation class (no badge).
 *
 * Human-readable copy lives in the i18n catalog under
 * `components.crew.toolMeta.*` and is resolved lazily (at call time, so the
 * active locale wins) by `toolMeta()` and `toolCategoryLabel()` — the table
 * itself holds only ids and classification, never translatable literals.
 */
import { i18nT } from '../../i18n/t'

export type ToolCategory = 'filesystem' | 'execution' | 'web' | 'kirocrew' | 'other'

export interface ToolMeta {
  label: string
  description: string
  category: ToolCategory
  /** true = read-only, false = can modify/run, undefined = unknown. */
  readOnly?: boolean
}

// Order the category groups render in.
export const TOOL_CATEGORY_ORDER: ToolCategory[] = ['filesystem', 'execution', 'web', 'kirocrew', 'other']

/** The display label for a category, resolved against the active locale. */
export function toolCategoryLabel(category: ToolCategory): string {
  return i18nT(`components.crew.toolMeta.category_${category}`)
}

// Static classification only: category + mutation class per tool id. The label
// and description are catalog keys resolved in `toolMeta()`.
const TOOL_CLASS: Record<string, { category: ToolCategory; readOnly?: boolean }> = {
  fs_read: { category: 'filesystem', readOnly: true },
  fs_write: { category: 'filesystem', readOnly: false },
  glob: { category: 'filesystem', readOnly: true },
  grep: { category: 'filesystem', readOnly: true },
  execute_bash: { category: 'execution', readOnly: false },
  code: { category: 'execution', readOnly: false },
  web_fetch: { category: 'web', readOnly: true },
  web_search: { category: 'web', readOnly: true },
  introspect: { category: 'kirocrew', readOnly: true },
  session: { category: 'kirocrew' },
  report: { category: 'kirocrew' },
  tool_search: { category: 'kirocrew', readOnly: true },
}

export function toolMeta(id: string): ToolMeta {
  const cls = TOOL_CLASS[id]
  if (!cls) return { label: id, description: '', category: 'other' }
  return {
    label: i18nT(`components.crew.toolMeta.${id}_label`),
    description: i18nT(`components.crew.toolMeta.${id}_description`),
    category: cls.category,
    readOnly: cls.readOnly,
  }
}
