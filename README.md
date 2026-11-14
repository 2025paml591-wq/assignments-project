# Cloud Resource & Cost Optimization Advisor

A read-only demonstration that compares five classical agent architectures with an Azure OpenAI/LangChain tool-calling agent. Every agent receives the same selected simulated scenario. The advisor does not connect to a cloud account or change resources.

## Features

- **Simple Reflex:** fixed condition-action rules on current metric values.
- **Model-Based Reflex:** estimates near-term pressure from recent CPU samples, peak, and trend.
- **Goal-Based:** searches candidate fleet sizes for the lowest-cost configuration meeting latency and availability goals.
- **Utility-Based:** scores configurations using explicit 50% latency, 30% availability, and 20% cost weights.
- **Learning:** uses persistent success/failure counts by strategy and supports recording observed outcomes in the UI.
- **Azure OpenAI LLM:** LangChain tool calling for workload metrics, cost, SLO targets, and scaling policy. The agent requires at least two distinct tool results before returning its structured, evidence-backed recommendation.
- **Scenarios:** A (critical CPU/latency), B (low-use/high-cost), C (sharp peaks and ambiguous daily average).

## Setup

Python 3.10+ recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Configure the Azure OpenAI endpoint, deployment, API version, and key in a root `.env` file:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_API_KEY=your-api-key
```

Keep `.env` local; it is excluded from Git so the API key is not committed.

## Run

```powershell
streamlit run app.py
```

Select A, B, or C. Run the classical agents for a deterministic comparison; run the Azure OpenAI agent separately because it requires a valid Azure endpoint and deployment. The UI displays configuration, estimated cost/performance, reasoning, tool evidence, and a monitoring plan. The learning-agent outcome controls update `agents/learning_experience.json`; rerun the classical agents to see updated history reflected in its reasoning.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover all five classical agents over all three scenarios, shared-input immutability, peak-aware model reasoning, learning-history updates, and scenario-specific outputs from each LLM tool. They do not make Azure OpenAI API calls.

## Demonstration results

The estimates below are illustrative and use the repository's fixed scenario rates and simple capacity model; they are not cloud-provider quotes or performance guarantees.

| Scenario | Approach | Example decision | Trade-off |
|---|---|---|---|
| A | Simple Reflex | Add two medium instances | Fast reaction to high CPU/latency; ignores demand history. |
| A | Model-Based | Add two medium instances | Uses sample trend and peak; forecast is a transparent heuristic. |
| A | Goal-Based | Select the best-effort fleet size in the configured range | Reports that no available candidate fully meets both SLOs; asks for a larger shape/architecture benchmark. |
| A | Utility-Based | Select the highest weighted score | Balances cost and service metrics; weights are subjective. |
| A | Learning | Scale out using historical success rate | Experience changes the scale-out magnitude; seeded counts are demonstration data. |
| B | Simple Reflex | Remove two instances and suggest smaller shape | Cost-focused rule may be too aggressive without a peak sample. |
| B | Goal-Based / Utility-Based | Search among smaller fleets | Explicitly evaluates SLO/cost trade-offs. |
| C | Simple Reflex | Add one instance on CPU peak | Reacts to peak but not duration/timing. |
| C | Model-Based / Learning | Keep warm capacity or use autoscaling | Recognizes samples/variability; costs more off-peak than a pure downsize. |
| C | Goal-Based / Utility-Based | Evaluate candidates against latency and availability | Result depends on estimated capacity-to-latency relationship and chosen weights. |

The model-based, goal-based, utility-based, and learning approaches can disagree because they use different information and objectives: current thresholds, sample-based state, explicit SLO constraints, weighted preferences, and past outcome frequencies respectively. All estimates are simplified; validate with real workload benchmarks, actual pricing, failure-domain design, and provider autoscaling behavior before acting.

## Screenshots and test output

Screenshots for A, B, and C are included in `screenshots/scenario-a.png`, `screenshots/scenario-b.png`, and `screenshots/scenario-c.png`. Test output and deterministic classical-agent results are recorded in `TEST_RESULTS.txt`.
