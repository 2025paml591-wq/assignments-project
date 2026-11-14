"""Azure OpenAI agent with LangChain tool calling and evidence-grounded output."""

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

from scenarios import get_scenario


load_dotenv()


@tool
def get_resource_metrics(scenario_id: str) -> str:
    """Get CPU, memory, latency, fleet size, and workload samples for a scenario."""
    return json.dumps(get_scenario(scenario_id)["metrics"], sort_keys=True)


@tool
def get_cost_estimate(scenario_id: str) -> str:
    """Estimate current monthly compute cost and per-instance hourly rate."""
    scenario = get_scenario(scenario_id)
    metrics = scenario["metrics"]
    monthly = metrics["current_instances"] * metrics["instance_hourly_cost_usd"] * 24 * 30
    return json.dumps({
        "current_monthly_cost_usd": round(monthly, 2),
        "instance_hourly_cost_usd": metrics["instance_hourly_cost_usd"],
        "current_instances": metrics["current_instances"],
    }, sort_keys=True)


@tool
def get_service_level_targets(scenario_id: str) -> str:
    """Get the scenario's p95 latency and availability goals."""
    return json.dumps(get_scenario(scenario_id)["targets"], sort_keys=True)


@tool
def get_scaling_policy(scenario_id: str) -> str:
    """Get the minimum, maximum, thresholds, and cooldown for scaling."""
    return json.dumps(get_scenario(scenario_id)["scaling_policy"], sort_keys=True)


TOOLS = [get_resource_metrics, get_cost_estimate, get_service_level_targets, get_scaling_policy]
TOOL_BY_NAME = {item.name: item for item in TOOLS}


class AzureOpenAIAdvisorAgent:
    name = "Azure OpenAI (tool-using)"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        missing = [
            name for name, value in (
                ("AZURE_OPENAI_DEPLOYMENT_NAME", self.model),
                ("AZURE_OPENAI_ENDPOINT", self.endpoint),
                ("AZURE_OPENAI_API_KEY", self.api_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing Azure OpenAI environment variables: {', '.join(missing)}")

    def decide(self, scenario: dict) -> dict:
        llm = AzureChatOpenAI(
            azure_deployment=self.model,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            temperature=1,
        ).bind_tools(TOOLS)
        system = SystemMessage(content=(
            "You are a read-only cloud capacity advisor. Never change resources. "
            "Call at least two distinct tools before recommending, preferably metrics and cost plus "
            "the relevant SLO/scaling tools. Use returned values as evidence, not guesses. "
            "After evidence is available, return ONLY a JSON object with keys: "
            "configuration (string), estimated_monthly_cost_usd (number), "
            "estimated_p95_latency_ms (integer), expected_availability_pct (number), "
            "recommendation (string), reasoning (array of strings), monitoring_plan (array of strings). "
            "Explain cost, performance, availability trade-offs and limitations."
        ))
        messages = [
            system,
            HumanMessage(content=(
                f"Advise on scenario {scenario['id']}: {scenario['name']}. "
                f"Scenario context: {json.dumps(scenario, sort_keys=True)}. "
                "Call get_resource_metrics and get_cost_estimate at minimum."
            )),
        ]
        evidence = {}
        tool_call_count = 0
        final_requested = False
        for _ in range(8):
            response = llm.invoke(messages)
            messages.append(response)
            calls = getattr(response, "tool_calls", []) or []
            if calls:
                for call in calls:
                    name = call.get("name")
                    if name not in TOOL_BY_NAME:
                        raise RuntimeError(f"Azure OpenAI requested an unsupported tool: {name!r}")
                    result = TOOL_BY_NAME[name].invoke(call.get("args", {}))
                    evidence[name] = result
                    tool_call_count += 1
                    messages.append(ToolMessage(content=str(result), tool_call_id=call.get("id", name)))
                if len(evidence) < 2:
                    missing = [name for name in ("get_resource_metrics", "get_cost_estimate") if name not in evidence]
                    messages.append(HumanMessage(content=f"Call another relevant tool now. Still required: {', '.join(missing) or 'a second distinct tool'}."))
                else:
                    messages.append(HumanMessage(content=(
                        "Using the tool results above, now return the requested JSON recommendation. "
                        "Cite at least two concrete returned values in reasoning."
                    )))
                    final_requested = True
                continue
            if len(evidence) < 2:
                messages.append(HumanMessage(content="You have not called two distinct tools yet. Call get_resource_metrics and get_cost_estimate before answering."))
                continue
            if not final_requested:
                messages.append(HumanMessage(content="Now return the requested JSON recommendation using the tool evidence; cite at least two concrete values."))
                final_requested = True
                continue
            payload = response.content
            if not isinstance(payload, str):
                raise RuntimeError("Azure OpenAI returned a non-text response instead of the required JSON.")
            try:
                answer = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Azure OpenAI did not return valid JSON: {exc}") from exc
            required = {
                "configuration", "estimated_monthly_cost_usd", "estimated_p95_latency_ms",
                "expected_availability_pct", "recommendation", "reasoning", "monitoring_plan",
            }
            missing = required.difference(answer)
            if missing:
                raise RuntimeError(f"Azure OpenAI JSON is missing required fields: {', '.join(sorted(missing))}")
            if not isinstance(answer["reasoning"], list) or not isinstance(answer["monitoring_plan"], list):
                raise RuntimeError("Azure OpenAI reasoning and monitoring_plan must be JSON arrays.")
            return {
                "agent": self.name,
                "configuration": str(answer["configuration"]),
                "estimated_monthly_cost_usd": float(answer["estimated_monthly_cost_usd"]),
                "estimated_p95_latency_ms": int(answer["estimated_p95_latency_ms"]),
                "expected_availability_pct": float(answer["expected_availability_pct"]),
                "recommendation": str(answer["recommendation"]),
                "reasoning": [str(item) for item in answer["reasoning"]],
                "monitoring_plan": [str(item) for item in answer["monitoring_plan"]],
                "evidence": [f"{name}: {value}" for name, value in evidence.items()],
                "strategy": f"LangChain tool calls ({tool_call_count} calls across {len(evidence)} distinct tools); model={self.model}",
            }
        raise RuntimeError(
            f"Azure OpenAI did not complete a tool-grounded recommendation after 8 turns "
            f"({tool_call_count} tool calls, {len(evidence)} distinct tools)."
        )
