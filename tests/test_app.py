import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from llm_agent import AzureOpenAIAdvisorAgent


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AdvisorUiTests(unittest.TestCase):
    def test_run_all_agents_reports_missing_azure_settings(self):
        environment = {
            "RUN_ALL_AGENTS": "true",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "",
            "AZURE_OPENAI_ENDPOINT": "",
            "AZURE_OPENAI_API_KEY": "",
        }
        with patch.dict(os.environ, environment):
            app = AppTest.from_file(str(APP_PATH)).run()
            next(button for button in app.button if button.label == "Run all six agents").click().run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 5)
        self.assertTrue(any("Azure OpenAI agent was not run" in item.value for item in app.warning))

    def test_run_all_agents_includes_azure_when_configured(self):
        azure_result = {
            "agent": AzureOpenAIAdvisorAgent.name,
            "configuration": "mocked",
            "estimated_monthly_cost_usd": 1,
            "estimated_p95_latency_ms": 1,
            "expected_availability_pct": 99.9,
            "recommendation": "mocked",
            "reasoning": ["mocked"],
            "evidence": ["mocked"],
            "monitoring_plan": ["mocked"],
        }
        environment = {
            "RUN_ALL_AGENTS": "true",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "",
            "AZURE_OPENAI_ENDPOINT": "",
            "AZURE_OPENAI_API_KEY": "",
        }
        received_settings = {}

        def fake_decide(agent, scenario):
            received_settings.update(
                model=agent.model,
                endpoint=agent.endpoint,
                api_key=agent.api_key,
            )
            return azure_result

        with patch.dict(os.environ, environment), patch.object(
            AzureOpenAIAdvisorAgent, "decide", fake_decide
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.text_input[0].set_value("test-deployment")
            app.text_input[1].set_value("https://example.invalid")
            app.text_input[2].set_value("not-a-real-key")
            next(button for button in app.button if button.label == "Run all six agents").click().run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 6)
        self.assertIn(AzureOpenAIAdvisorAgent.name, app.dataframe[0].value["Agent"].tolist())
        self.assertEqual(
            received_settings,
            {
                "model": "test-deployment",
                "endpoint": "https://example.invalid",
                "api_key": "not-a-real-key",
            },
        )

    def test_run_all_agents_setting_can_disable_azure(self):
        environment = {
            "RUN_ALL_AGENTS": "false",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
            "AZURE_OPENAI_ENDPOINT": "https://example.invalid",
            "AZURE_OPENAI_API_KEY": "not-a-real-key",
        }
        with patch.dict(os.environ, environment), patch.object(
            AzureOpenAIAdvisorAgent, "decide", side_effect=AssertionError("Azure agent should be disabled")
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            next(button for button in app.button if button.label == "Run five classical agents").click().run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 5)


if __name__ == "__main__":
    unittest.main()
