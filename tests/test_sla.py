from orchestrator.sla import SLA, SLARegistry


def test_sla_renegotiation():
    registry = SLARegistry()
    registry.register("providerA", SLA(latency_ms=100, cost_per_token=0.01))

    def proposal(current: SLA) -> SLA:
        return SLA(latency_ms=current.latency_ms + 50, cost_per_token=current.cost_per_token)

    updated = registry.renegotiate("providerA", proposal)
    assert updated.latency_ms == 150
    assert updated.cost_per_token == 0.01
