import tempfile
import unittest
from pathlib import Path

from agents import (
    GoalBasedAgent,
    LearningAgent,
    ModelBasedReflexAgent,
    SimpleReflexAgent,
    UtilityBasedAgent,
)
from llm_agent import get_cost_estimate, get_resource_metrics, get_scaling_policy, get_service_level_targets
from scenarios import SCENARIOS, get_scenario


class ClassicalAgentTests(unittest.TestCase):
    def setUp(self):
        self.agents = [
            SimpleReflexAgent(),
            ModelBasedReflexAgent(),
            GoalBasedAgent(),
            UtilityBasedAgent(),
            LearningAgent(),
        ]

    def test_all_classical_agents_decide_for_each_shared_scenario(self):
        for scenario_id in SCENARIOS:
            scenario = get_scenario(scenario_id)
            original = get_scenario(scenario_id)
            for agent in self.agents:
                decision = agent.decide(scenario)
                self.assertTrue(decision.agent)
                self.assertGreater(decision.estimated_monthly_cost_usd, 0)
                self.assertTrue(decision.reasoning)
                self.assertTrue(decision.evidence)
            self.assertEqual(scenario, original, "Agents must not mutate shared scenario data.")

    def test_model_based_agent_responds_to_peak_variability(self):
        scenario = get_scenario("C")
        decision = ModelBasedReflexAgent().decide(scenario)
        self.assertIn("samples", " ".join(decision.reasoning))
        self.assertGreater(int(decision.configuration.split()[0]), scenario["metrics"]["current_instances"])

    def test_goal_agent_accepts_exact_availability_threshold(self):
        decision = GoalBasedAgent().decide(get_scenario("B"))
        self.assertIn("2 x", decision.configuration)
        self.assertGreaterEqual(decision.expected_availability_pct, 99.9)

    def test_model_based_agent_does_not_scale_up_for_small_low_load_fluctuations(self):
        scenario = get_scenario("B")
        decision = ModelBasedReflexAgent().decide(scenario)
        self.assertLessEqual(int(decision.configuration.split()[0]), scenario["metrics"]["current_instances"])

    def test_learning_agent_uses_and_updates_persistent_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = LearningAgent(Path(temp_dir) / "experience.json")
            scenario = get_scenario("A")
            before = agent.decide(scenario)
            agent.record_outcome(before.strategy, False)
            after = agent.decide(scenario)
            self.assertTrue((Path(temp_dir) / "experience.json").exists())
            self.assertNotEqual(before.reasoning[1], after.reasoning[1])

    def test_scenario_copy_is_isolated(self):
        scenario = get_scenario("A")
        scenario["metrics"]["cpu_avg_pct"] = 0
        self.assertNotEqual(get_scenario("A")["metrics"]["cpu_avg_pct"], 0)


class ToolTests(unittest.TestCase):
    def test_tool_evidence_is_scenario_specific(self):
        metrics = get_resource_metrics.invoke({"scenario_id": "A"})
        cost = get_cost_estimate.invoke({"scenario_id": "B"})
        targets = get_service_level_targets.invoke({"scenario_id": "C"})
        policy = get_scaling_policy.invoke({"scenario_id": "C"})
        self.assertIn('"cpu_avg_pct": 91', metrics)
        self.assertIn('"current_monthly_cost_usd": 921.6', cost)
        self.assertIn('"p95_latency_ms": 350', targets)
        self.assertIn('"max_instances": 8', policy)


if __name__ == "__main__":
    unittest.main()
