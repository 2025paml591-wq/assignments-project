"""A reflex agent that updates an internal workload estimate from recent samples."""

from statistics import mean

from .base import make_decision


class ModelBasedReflexAgent:
    name = "Model-Based Reflex"

    def decide(self, scenario: dict):
        m = scenario["metrics"]
        policy = scenario["scaling_policy"]
        samples = m["cpu_samples_pct"]
        recent_mean = mean(samples)
        rising = len(samples) > 1 and samples[-1] - samples[0] >= 15
        # Estimate near-term pressure from both the trailing samples and observed peak.
        predicted_cpu = max(recent_mean, m["cpu_peak_pct"] * 0.8)
        if predicted_cpu >= policy["scale_out_cpu_pct"] or rising:
            instances = min(m["current_instances"] + 2, policy["max_instances"])
            estimated_latency = max(
                scenario["targets"]["p95_latency_ms"],
                int(m["p95_latency_ms"] * (0.65 if m["cpu_avg_pct"] > 75 else 0.78)),
            )
            reason = (
                f"Internal state estimate uses {len(samples)} CPU samples (mean {recent_mean:.1f}%) "
                f"and peak-based forecast {predicted_cpu:.1f}%; "
                f"{'the series is rising' if rising else 'the series is not rising'}."
            )
            recommendation = "Scale out by two instances to cover the estimated near-term demand."
        else:
            instances = max(m["current_instances"] - 1, policy["min_instances"])
            estimated_latency = max(scenario["targets"]["p95_latency_ms"], int(m["p95_latency_ms"] * 1.2))
            reason = (
                f"Recent CPU sample mean is {recent_mean:.1f}% and predicted pressure is "
                f"{predicted_cpu:.1f}%, below the scale-out threshold."
            )
            recommendation = "Scale in by one instance while retaining the configured minimum."
        return make_decision(
            scenario, agent=self.name, instances=instances,
            configuration=f"{instances} x {m['instance_type']} (model-estimated)",
            latency_ms=estimated_latency,
            availability_pct=99.95 if instances > m["current_instances"] else 99.9,
            recommendation=recommendation,
            reasoning=[
                reason,
                f"Scale-out trigger is {policy['scale_out_cpu_pct']}%; cooldown is {policy['cooldown_seconds']} seconds.",
                "The internal estimate is a simple heuristic, not a trained demand forecast.",
            ],
            strategy="Update estimated load from sample mean, peak, and direction of change.",
        )
