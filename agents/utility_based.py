"""A weighted-utility agent that makes cost/performance trade-offs explicit."""

from .base import make_decision


class UtilityBasedAgent:
    name = "Utility-Based"

    def decide(self, scenario: dict):
        m = scenario["metrics"]
        policy = scenario["scaling_policy"]
        target = scenario["targets"]["p95_latency_ms"]
        candidates = []
        for instances in range(policy["min_instances"], policy["max_instances"] + 1):
            ratio = instances / max(1, m["current_instances"])
            latency = max(80, int(m["p95_latency_ms"] / ratio))
            availability = round(min(99.99, 99.8 + instances * 0.05), 2)
            cost = instances * m["instance_hourly_cost_usd"] * 24 * 30
            latency_score = max(0.0, 1 - max(0, latency - target) / max(target, 1))
            availability_score = min(1.0, availability / scenario["targets"]["availability_pct"])
            cost_score = 1 - (cost / max(1, policy["max_instances"] * m["instance_hourly_cost_usd"] * 24 * 30))
            utility = 0.50 * latency_score + 0.30 * availability_score + 0.20 * cost_score
            candidates.append((utility, instances, latency, availability, cost, latency_score, availability_score, cost_score))
        best = max(candidates, key=lambda item: (item[0], -item[1]))
        utility, instances, latency, availability, cost, latency_score, availability_score, cost_score = best
        return make_decision(
            scenario, agent=self.name, instances=instances,
            configuration=f"{instances} x {m['instance_type']} (highest utility)",
            latency_ms=latency,
            availability_pct=availability,
            recommendation="Choose the candidate with the highest weighted utility, rather than optimizing a single metric.",
            reasoning=[
                f"Best utility is {utility:.3f}: latency={latency_score:.2f}, availability={availability_score:.2f}, cost={cost_score:.2f}.",
                "Weights are 50% latency, 30% availability, and 20% cost; changing these weights can change the choice.",
                f"The selected candidate is estimated at ${cost:.2f}/month.",
            ],
            strategy="Score each candidate: 0.50*latency + 0.30*availability + 0.20*cost.",
        )
