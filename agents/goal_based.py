"""A goal-directed agent that searches candidate sizes against SLO constraints."""

from .base import make_decision


class GoalBasedAgent:
    name = "Goal-Based"

    def decide(self, scenario: dict):
        m = scenario["metrics"]
        policy = scenario["scaling_policy"]
        target = scenario["targets"]["p95_latency_ms"]
        candidates = []
        for instances in range(policy["min_instances"], policy["max_instances"] + 1):
            capacity_ratio = instances / max(1, m["current_instances"])
            latency = max(80, int(m["p95_latency_ms"] / capacity_ratio))
            availability = round(min(99.99, 99.8 + instances * 0.05), 2)
            meets_slo = latency <= target and availability >= scenario["targets"]["availability_pct"]
            candidates.append((instances, latency, availability, meets_slo))
        feasible = [candidate for candidate in candidates if candidate[3]]
        if feasible:
            instances, latency, availability, _ = min(feasible, key=lambda item: item[0])
            rationale = f"Selected the least-capacity candidate that meets latency and availability goals ({latency} ms, {availability:.2f}%)."
            recommendation = "Choose the lowest-cost candidate that satisfies the stated latency and availability goals."
        else:
            instances, latency, availability, _ = min(
                candidates,
                key=lambda item: (
                    max(0, item[1] - target) + max(0, scenario["targets"]["availability_pct"] - item[2]) * 100,
                    item[0],
                ),
            )
            rationale = "No candidate in the configured range meets both SLOs; selected the candidate with the smallest estimated SLO shortfall."
            recommendation = (
                "The configured fleet range cannot meet both SLOs; use the least-shortfall candidate "
                "only as a starting point, then benchmark a larger instance shape or architecture."
            )
        return make_decision(
            scenario, agent=self.name, instances=instances,
            configuration=f"{instances} x {m['instance_type']} (goal search)",
            latency_ms=latency,
            availability_pct=availability,
            recommendation=recommendation,
            reasoning=[
                rationale,
                f"Goals: p95 <= {target} ms and availability >= {scenario['targets']['availability_pct']}%.",
                f"Evaluated {len(candidates)} candidates from {policy['min_instances']} to {policy['max_instances']} instances.",
            ],
            strategy="Enumerate configurations and select the cheapest feasible SLO candidate.",
        )
