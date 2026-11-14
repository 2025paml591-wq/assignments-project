"""Shared, simulated workload scenarios used by every agent."""

from copy import deepcopy


SCENARIOS = {
    "A": {
        "id": "A",
        "name": "High CPU and slow responses",
        "category": "Critical / failure",
        "description": "CPU is consistently saturated and p95 latency is above the service target.",
        "metrics": {
            "cpu_avg_pct": 91,
            "cpu_peak_pct": 98,
            "memory_avg_pct": 76,
            "p95_latency_ms": 840,
            "cpu_samples_pct": [79, 83, 89, 92],
            "traffic_variability": 1.15,
            "current_instances": 3,
            "instance_type": "standard.medium",
            "instance_hourly_cost_usd": 0.08,
        },
        "targets": {"p95_latency_ms": 300, "availability_pct": 99.9},
        "scaling_policy": {
            "min_instances": 2,
            "max_instances": 8,
            "scale_out_cpu_pct": 70,
            "scale_in_cpu_pct": 35,
            "cooldown_seconds": 300,
        },
    },
    "B": {
        "id": "B",
        "name": "Low utilization, high spend",
        "category": "Normal / cost optimization",
        "description": "CPU and memory are low, latency is healthy, and the current shape is expensive.",
        "metrics": {
            "cpu_avg_pct": 18,
            "cpu_peak_pct": 29,
            "memory_avg_pct": 24,
            "p95_latency_ms": 120,
            "cpu_samples_pct": [16, 20, 18, 19],
            "traffic_variability": 1.1,
            "current_instances": 4,
            "instance_type": "standard.xlarge",
            "instance_hourly_cost_usd": 0.32,
        },
        "targets": {"p95_latency_ms": 300, "availability_pct": 99.9},
        "scaling_policy": {
            "min_instances": 2,
            "max_instances": 8,
            "scale_out_cpu_pct": 70,
            "scale_in_cpu_pct": 35,
            "cooldown_seconds": 300,
        },
    },
    "C": {
        "id": "C",
        "name": "Sharp peak/off-peak swings",
        "category": "Edge / ambiguous",
        "description": "Daily averages look comfortable, but sharp peaks can breach latency and availability targets.",
        "metrics": {
            "cpu_avg_pct": 55,
            "cpu_peak_pct": 94,
            "memory_avg_pct": 48,
            "p95_latency_ms": 460,
            "cpu_samples_pct": [20, 86, 24, 94],
            "traffic_variability": 4.7,
            "current_instances": 3,
            "instance_type": "standard.medium",
            "instance_hourly_cost_usd": 0.08,
        },
        "targets": {"p95_latency_ms": 350, "availability_pct": 99.9},
        "scaling_policy": {
            "min_instances": 2,
            "max_instances": 8,
            "scale_out_cpu_pct": 70,
            "scale_in_cpu_pct": 35,
            "cooldown_seconds": 300,
        },
    },
}


def get_scenario(scenario_id: str) -> dict:
    """Return a fresh copy so an agent cannot mutate shared scenario data."""
    try:
        return deepcopy(SCENARIOS[scenario_id])
    except KeyError as exc:
        raise ValueError(f"Unknown scenario: {scenario_id!r}") from exc


def monthly_cost(scenario: dict, instances: int, hourly_rate: float | None = None) -> float:
    rate = hourly_rate if hourly_rate is not None else scenario["metrics"]["instance_hourly_cost_usd"]
    return round(instances * rate * 24 * 30, 2)
