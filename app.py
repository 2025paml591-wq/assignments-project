"""Streamlit comparison UI for simulated cloud resource recommendations."""

import os

import pandas as pd
import streamlit as st

from agents import (
    GoalBasedAgent,
    LearningAgent,
    ModelBasedReflexAgent,
    SimpleReflexAgent,
    UtilityBasedAgent,
)
from llm_agent import AzureOpenAIAdvisorAgent
from scenarios import SCENARIOS, get_scenario


st.set_page_config(page_title="Cloud Resource Advisor", page_icon="☁️", layout="wide")
st.title("Cloud Resource & Cost Optimization Advisor")
st.caption("Compare six decision approaches on the same simulated workload. Recommendations are advisory only; no cloud resources are modified.")

scenario_id = st.selectbox(
    "Select workload scenario",
    list(SCENARIOS),
    format_func=lambda key: f"{key} — {SCENARIOS[key]['name']} ({SCENARIOS[key]['category']})",
)
scenario = get_scenario(scenario_id)
m = scenario["metrics"]
st.subheader(scenario["name"])
st.write(scenario["description"])
metric_cols = st.columns(5)
metric_cols[0].metric("CPU average / peak", f"{m['cpu_avg_pct']}% / {m['cpu_peak_pct']}%")
metric_cols[1].metric("Memory average", f"{m['memory_avg_pct']}%")
metric_cols[2].metric("p95 latency", f"{m['p95_latency_ms']} ms")
metric_cols[3].metric("Current fleet", f"{m['current_instances']} × {m['instance_type']}")
metric_cols[4].metric("Current monthly cost", f"${m['current_instances'] * m['instance_hourly_cost_usd'] * 24 * 30:,.2f}")

with st.expander("Scenario data shared by all agents"):
    st.json(scenario)

if "decisions" not in st.session_state or st.session_state.get("scenario_id") != scenario_id:
    st.session_state["decisions"] = {}
    st.session_state["scenario_id"] = scenario_id

classical_agents = [
    SimpleReflexAgent(),
    ModelBasedReflexAgent(),
    GoalBasedAgent(),
    UtilityBasedAgent(),
]
learning_agent = LearningAgent()
run_classical = st.button("Run / refresh five classical agents", type="primary")
if run_classical:
    st.session_state["decisions"] = {
        agent.name: agent.decide(scenario).as_dict() for agent in classical_agents
    }
    st.session_state["decisions"][learning_agent.name] = learning_agent.decide(scenario).as_dict()

col_a, col_b = st.columns([1, 2])
with col_a:
    azure_deployment = st.text_input(
        "Azure OpenAI deployment",
        value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
    )
    run_llm = st.button("Run Azure OpenAI agent")
with col_b:
    st.info("The Azure OpenAI agent calls metrics and cost tools (and may call SLO/scaling tools). Set its endpoint, deployment, and API key in the root .env file.")
if run_llm:
    with st.spinner("Calling Azure OpenAI and collecting tool evidence..."):
        llm_result = AzureOpenAIAdvisorAgent(model=azure_deployment).decide(scenario)
    st.session_state["decisions"][llm_result["agent"]] = llm_result

decisions = st.session_state["decisions"]
if decisions:
    st.subheader("Comparison")
    comparison = [{
        "Agent": name,
        "Configuration": result["configuration"],
        "Monthly cost (est.)": f"${result['estimated_monthly_cost_usd']:,.2f}",
        "p95 latency (est.)": f"{result['estimated_p95_latency_ms']} ms",
        "Availability (est.)": f"{result['expected_availability_pct']:.2f}%",
    } for name, result in decisions.items()]
    st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
    for name, result in decisions.items():
        with st.expander(f"{name}: {result['recommendation']}", expanded=True):
            st.markdown(f"**Candidate:** {result['configuration']}")
            st.markdown(
                f"**Estimates:** ${result['estimated_monthly_cost_usd']:,.2f}/month · "
                f"{result['estimated_p95_latency_ms']} ms p95 · "
                f"{result['expected_availability_pct']:.2f}% availability"
            )
            st.markdown("**Why this differs**")
            for item in result["reasoning"]:
                st.markdown(f"- {item}")
            st.markdown("**Evidence**")
            for item in result["evidence"]:
                st.markdown(f"- {item}")
            st.markdown("**Monitoring plan**")
            for item in result["monitoring_plan"]:
                st.markdown(f"- {item}")

    if learning_agent.name in decisions:
        st.subheader("Update Learning Agent experience")
        learned = decisions[learning_agent.name]
        st.write(f"Current strategy: **{learned['strategy']}**. Record an observed outcome to update the persistent experience file.")
        outcome_col1, outcome_col2 = st.columns(2)
        with outcome_col1:
            if st.button("Record success"):
                learning_agent.record_outcome(learned["strategy"], True)
                st.success("Recorded success. Rerun the classical agents to use the updated experience.")
        with outcome_col2:
            if st.button("Record failure"):
                learning_agent.record_outcome(learned["strategy"], False)
                st.warning("Recorded failure. Rerun the classical agents to use the updated experience.")
else:
    st.write("Run the classical agents to see their recommendations. The Azure OpenAI agent can be run separately.")

st.caption("Estimates are illustrative, based on the simulated scenario and simple capacity heuristics. Validate with workload benchmarks and provider pricing before making any change.")
