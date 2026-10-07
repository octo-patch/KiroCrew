import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import AgentSelector from '../components/AgentSelector'
import type { KiroCrewAgent } from '../components/AgentSelector'

/**
 * A shadowed collision row explains the override on its detail line: which
 * same-named definition runs and where it lives. The name sits mid-sentence
 * rather than leading it, because the row title directly above already shows it.
 */
describe('AgentSelector override-detail line names the definition that runs', () => {
  const collision: KiroCrewAgent[] = [
    {
      name: 'reviewer', kiro_agent: '', workspace: 'default', memory_store: 'default',
      description: '', source: 'kirocrew', scope: 'project',
    },
  ]

  it("names this project's definition as the one that runs", () => {
    render(
      <AgentSelector
        agents={collision}
        defaultAgent="default"
        value=""
        onChange={() => {}}
        shadowedGlobals={new Set(['reviewer'])}
      />,
    )
    fireEvent.click(screen.getByLabelText('Switch agent'))
    const row = screen.getByRole('option', { name: /reviewer/ })
    // A shadowed row shows the full explanation as body text, not as a native
    // `title` on the chip: a tooltip has no keyboard path and no hover on
    // touch, and this sentence is the only thing telling a user why their
    // configured agent is not the one running. It is also the string every
    // catalog already translates, so non-English users get the same reassurance.
    expect(row).toHaveTextContent(
      "This job runs the reviewer from this project's .kiro/agents/, not your global one.",
    )
    // The chip is warn-toned so it does not read as one more source label.
    expect(screen.getByText('replaces global')).toHaveClass('text-warn')
  })
})
