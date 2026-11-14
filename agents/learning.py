"""Experience-based agent with explicit, persistent outcome feedback."""

import json
from pathlib import Path

from .base import make_decision


DEFAULT_EXPERIENCE = {
    "scale_out": {"success": 8, "failure": 1},
    "rightsize": {"success": 5, "failure": 3},
    "autoscale": {"success": 7, "failure": 1},
}


class LearningAgent:
    name = "Learning"

    def __init__(self, experience_path: str | Path | None = None):
        self.experience_path = Path(experience_path) if experience_path else Path(__file__).with_name("learning_experience.json")

    def _load_experience(self) -> dict:
        if not self.experience_path.exists():
            return json.loads(json.dumps(DEFAULT_EXPERIENCE))
        with self.experience_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        for strategy in DEFAULT_EXPERIENCE:
            counts = data.setdefault(strategy, {})
            counts.setdefault("success", 0)
            counts.setdefault("failure", 0)
        return data

    def record_outcome(self, strategy: str, success: bool) -> dict:
        if strategy not in DEFAULT_EXPERIENCE:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        data = self._load_experience()
        outcome = "success" if success else "failure"
        data[strategy][outcome] += 1
        self.experience_path.parent.mkdir(parents=True, exist_ok=True)
        with self.experience_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
        return data

    def decide(self, scenario: dict):
        m = scenario["metrics"]
        data = self._load_experience()
        variability = m["traffic_variability"]
        if m["cpu_avg_pct"] >= 80 or m["p95_latency_ms"] > scenario["targets"]["p95_latency_ms"]:
            strategy = "scale_out"
        elif variability >= 3 or m["cpu_peak_pct"] >= 85:
            strategy = "autoscale"
        else:
            strategy = "rightsize"
        counts = data[strategy]
        total = counts["success"] + counts["failure"]
        rate = counts["success"] / total if total else 0.5
        current = m["current_instances"]
        if strategy == "scale_out":
            instances = min(current + (2 if rate >= 0.7 else 1), scenario["scaling_policy"]["max_instances"])
            config = f"{instances} x {m['instance_type']} (experience-weighted scale-out)"
            latency = max(scenario["targets"]["p95_latency_ms"], int(m["p95_latency_ms"] * (0.62 if rate >= 0.7 else 0.76)))
            recommendation = "Scale out, with the size of the change moderated by recorded outcome success."
        elif strategy == "autoscale":
            instances = min(current + 2, scenario["scaling_policy"]["max_instances"])
            config = f"{instances} x {m['instance_type']} with scheduled/policy autoscaling"
            latency = max(scenario["targets"]["p95_latency_ms"], int(m["p95_latency_ms"] * 0.72))
            recommendation = "Use autoscaling with a warm minimum to handle sharp peaks without paying for peak capacity all day."
        else:
            instances = max(current - 1, scenario["scaling_policy"]["min_instances"])
            config = f"{instances} x smaller instance (experience-weighted rightsize)"
            latency = max(scenario["targets"]["p95_latency_ms"], int(m["p95_latency_ms"] * 1.2))
            recommendation = "Rightsize conservatively because current utilization is low."
        return make_decision(
            scenario, agent=self.name, instances=instances,
            configuration=config,
            latency_ms=latency,
            availability_pct=99.95 if instances >= current else 99.9,
            recommendation=recommendation,
            reasoning=[
                f"Selected {strategy} from the current workload signals.",
                f"Experience for {strategy}: {counts['success']} successes / {counts['failure']} failures "
                f"({rate:.0%} observed success rate across {total} outcomes).",
                "For scale-out, a stronger success history increases the change magnitude; the selected strategy's history is reported above.",
            ],
            strategy=strategy,
            hourly_rate=m["instance_hourly_cost_usd"] * (0.5 if strategy == "rightsize" else 1.0),
        )
