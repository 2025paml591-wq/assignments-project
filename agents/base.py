"""Shared output contract and candidate calculations for classical agents."""

from dataclasses import asdict, dataclass, field

from scenarios import monthly_cost


@dataclass
class Decision:
    agent: str
    configuration: str
    estimated_monthly_cost_usd: float
    estimated_p95_latency_ms: int
    expected_availability_pct: float
    recommendation: str
    reasoning: list[str] = field(default_factory=list)
    monitoring_plan: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    strategy: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def current_evidence(scenario: dict) -> list[str]:
    m = scenario["metrics"]
    t = scenario["targets"]
    return [
        f"Observed CPU avg/peak: {m['cpu_avg_pct']}% / {m['cpu_peak_pct']}%.",
        f"Observed memory average: {m['memory_avg_pct']}%.",
        f"Observed p95 latency: {m['p95_latency_ms']} ms; target: {t['p95_latency_ms']} ms.",
        f"Current fleet: {m['current_instances']} x {m['instance_type']}.",
    ]


def make_decision(
    scenario: dict,
    *,
    agent: str,
    instances: int,
    configuration: str,
    latency_ms: int,
    availability_pct: float,
    recommendation: str,
    reasoning: list[str],
    strategy: str,
    hourly_rate: float | None = None,
    monitoring_plan: list[str] | None = None,
) -> Decision:
    return Decision(
        agent=agent,
        configuration=configuration,
        estimated_monthly_cost_usd=monthly_cost(scenario, instances, hourly_rate),
        estimated_p95_latency_ms=latency_ms,
        expected_availability_pct=availability_pct,
        recommendation=recommendation,
        reasoning=reasoning,
        monitoring_plan=monitoring_plan or [
            "Track p95 latency, CPU, memory, and availability at 5-minute intervals.",
            "Review the recommendation after one representative peak/off-peak cycle.",
            "Roll out changes gradually and keep a rollback path; no resources are changed by this advisor.",
        ],
        evidence=current_evidence(scenario),
        strategy=strategy,
    )
