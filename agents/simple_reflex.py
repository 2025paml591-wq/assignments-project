"""A condition-action agent that reacts only to current observed metrics."""

from .base import make_decision


class SimpleReflexAgent:
    name = "Simple Reflex"

    def decide(self, scenario: dict):
        m = scenario["metrics"]
        target = scenario["targets"]["p95_latency_ms"]
        current = m["current_instances"]
        if m["cpu_avg_pct"] >= 80 or m["p95_latency_ms"] > target:
            instances = min(current + 2, scenario["scaling_policy"]["max_instances"])
            return make_decision(
                scenario, agent=self.name, instances=instances,
                configuration=f"{instances} x {m['instance_type']} (scale out)",
                latency_ms=max(target, int(m["p95_latency_ms"] * 0.62)),
                availability_pct=99.95,
                recommendation="Scale out immediately by two instances to relieve sustained CPU and latency pressure.",
                reasoning=[
                    f"Rule fired: CPU average is {m['cpu_avg_pct']}% or p95 latency exceeds its {target} ms target.",
                    "This agent reacts to the current reading only; it does not use history or compare alternative objectives.",
                ],
                strategy="IF CPU >= 80% OR p95 latency > target, add two instances.",
            )
        if m["cpu_avg_pct"] < 30 and m["memory_avg_pct"] < 40:
            instances = max(current - 2, scenario["scaling_policy"]["min_instances"])
            return make_decision(
                scenario, agent=self.name, instances=instances,
                configuration=f"{instances} x smaller instance (scale down)",
                latency_ms=max(target, int(m["p95_latency_ms"] * 1.35)),
                availability_pct=99.9,
                recommendation="Downsize two instances because current CPU and memory are low.",
                reasoning=[
                    f"Rule fired: CPU is {m['cpu_avg_pct']}% and memory is {m['memory_avg_pct']}%.",
                    "This direct rule does not account for the range of recent workload samples.",
                ],
                strategy="IF CPU < 30% AND memory < 40%, remove two instances.",
                hourly_rate=m["instance_hourly_cost_usd"] * 0.5,
            )
        instances = current + 1 if m["cpu_peak_pct"] >= 85 else current
        instances = min(instances, scenario["scaling_policy"]["max_instances"])
        return make_decision(
            scenario, agent=self.name, instances=instances,
            configuration=f"{instances} x {m['instance_type']} (reactive capacity)",
            latency_ms=max(target, int(m["p95_latency_ms"] * 0.8)),
            availability_pct=99.92,
            recommendation="Add one instance because the observed CPU peak is high.",
            reasoning=[
                f"CPU average is {m['cpu_avg_pct']}%, but peak CPU reached {m['cpu_peak_pct']}%.",
                "The fixed peak threshold triggers a small scale-out without forecasting.",
            ],
            strategy="IF CPU peak >= 85%, add one instance.",
        )
