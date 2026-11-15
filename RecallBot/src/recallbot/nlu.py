import re
from dataclasses import dataclass, field

from .retrieval import normalize_query


INTENT_PATTERNS = {
    "forget": re.compile(r"\b(forget|delete|remove|bhool)\b", re.I),
    "confirm": re.compile(r"\b(yes|confirm|confirmed|go ahead|save it|do it|haan)\b", re.I),
    "cancel": re.compile(r"\b(cancel|no,? don't|stop)\b", re.I),
    "goal_create": re.compile(
        r"\b(?:create|set up|start|open)\b.{0,40}\b(?:goal|fund)\b|"
        r"\b(?:goal|fund)\b.{0,25}\b(?:create|set up|start)\b|"
        r"\b(?:set aside|bacha?t)\b",
        re.I,
    ),
    "episodic_recall": re.compile(r"\b(what happened|last time|previous session|booked|episode)\b", re.I),
    "semantic_recall": re.compile(r"\b(what (?:is|was) my|what did i say|remember my|which city|my preference)\b", re.I),
    "budget_lookup": re.compile(r"\b(balance|budget|spend|income|rent|expense|category)\b", re.I),
    "knowledge_query": re.compile(
        r"\b(policy|guide|rule|code|fee|window|limit|how do i|what does)\b|"
        r"\b[A-Z]{2,}-\d+\b",
        re.I,
    ),
    "thanks": re.compile(r"\b(thanks|thank you|shukriya)\b", re.I),
}

AMOUNT = re.compile(r"(?:₹|rs\.?\s*)?([\d,]+(?:\.\d{1,2})?)\s*(?:rupees?|rs\.?)?", re.I)
DATE = re.compile(r"\b(20\d{2}-\d{2}-\d{2}|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4})\b", re.I)
FACTS = {
    "name": re.compile(r"\b(?:my name is|mera naam)\s+([A-Za-z][A-Za-z -]+)", re.I),
    "city": re.compile(r"\b(?:live|based|move)\s+(?:in|to)\s+([A-Z][A-Za-z -]+)", re.I),
    "income": re.compile(r"\b(?:take.home income|monthly income|salary|aamdani)\D{0,30}(?:₹|rs\.?\s*)?([\d,]+)", re.I),
    "rent": re.compile(r"\b(?:rent|kiraya)\D{0,25}(?:₹|rs\.?\s*)?([\d,]+)", re.I),
    "grocery_budget": re.compile(r"\b(?:groceries|grocery|food)\D{0,30}(?:₹|rs\.?\s*)?([\d,]+)", re.I),
    "family_support": re.compile(r"\b(?:support|send|help)\D{0,30}(?:parents|family)\D{0,30}(?:₹|rs\.?\s*)?([\d,]+)", re.I),
    "preference": re.compile(r"\b(?:prefer|preference is|i like)\s+(.+?)(?:[.!?]|$)", re.I),
    "payday": re.compile(r"\bpaid\s+(?:on|every)\s+([A-Za-z0-9 -]+?)(?:[.!?]|$)", re.I),
    "savings_goal": re.compile(r"\b(?:savings goal|goal is)\D{0,25}(?:₹|rs\.?\s*)?([\d,]+)", re.I),
    "dining_cap": re.compile(r"\b(?:dining|eating out)\D{0,30}(?:₹|rs\.?\s*)?([\d,]+)", re.I),
}


@dataclass
class NLUResult:
    intent: str
    normalized: str
    entities: dict[str, str] = field(default_factory=dict)


def parse_amount(text: str) -> float | None:
    without_dates = re.sub(r"\b20\d{2}-\d{2}-\d{2}\b", "", text)
    matches = AMOUNT.findall(without_dates)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def classify(text: str) -> NLUResult:
    normalized = normalize_query(text)
    intent = "open"
    for candidate in (
        "forget",
        "confirm",
        "cancel",
        "goal_create",
        "episodic_recall",
        "semantic_recall",
        "knowledge_query",
        "budget_lookup",
        "thanks",
    ):
        if INTENT_PATTERNS[candidate].search(normalized):
            intent = candidate
            break
    entities: dict[str, str] = {}
    amount = parse_amount(text)
    if amount is not None:
        entities["amount"] = str(amount)
    date_match = DATE.search(text)
    if date_match:
        entities["date"] = date_match.group(0)
    for key, pattern in FACTS.items():
        match = pattern.search(text)
        if match:
            entities[key] = " ".join(match.group(1).split()).strip(" .")
    if text.strip().lower().startswith(("and for ", "what about ", "and on ")):
        entities["state_followup"] = "true"
    return NLUResult(intent=intent, normalized=normalized, entities=entities)
