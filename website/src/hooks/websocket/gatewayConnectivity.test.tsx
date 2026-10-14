/** The socket layer's connect/disconnect helpers move the store flag AND the
 *  errorReport seam the "Ask the agent" button reads, so the button follows
 *  the real socket state end to end. */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { act, render, screen } from '@testing-library/react'
import AskAgentButton from '../../components/AskAgentButton'
import { sseConnected, sseDisconnected } from '../../store/dashboardSlice'
import { __resetGatewayConnectedForTests, isGatewayConnected } from '../../utils/errorReport'
import { markGatewayConnected, markGatewayDisconnected } from './gatewayConnectivity'

beforeEach(() => { __resetGatewayConnectedForTests() })
afterEach(() => { __resetGatewayConnectedForTests() })

describe('gatewayConnectivity helpers', () => {
  it('disconnect dispatches sseDisconnected and hides the hand-off; connect restores both', () => {
    const dispatch = vi.fn()
    render(<AskAgentButton message="Failed to fetch" />)
    expect(screen.getByRole('button', { name: /ask the agent/i })).toBeInTheDocument()

    act(() => markGatewayDisconnected(dispatch as never))
    expect(dispatch).toHaveBeenLastCalledWith(sseDisconnected())
    expect(isGatewayConnected()).toBe(false)
    expect(screen.queryByRole('button', { name: /ask the agent/i })).not.toBeInTheDocument()

    act(() => markGatewayConnected(dispatch as never))
    expect(dispatch).toHaveBeenLastCalledWith(sseConnected())
    expect(isGatewayConnected()).toBe(true)
    expect(screen.getByRole('button', { name: /ask the agent/i })).toBeInTheDocument()
  })
})
