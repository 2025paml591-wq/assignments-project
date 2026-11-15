from pathlib import Path

from recallbot.config import Paths


COMMON = (
    "This note is educational and is not a promise of a particular financial outcome. "
    "Use the household's own statements and verified account records; do not infer a balance "
    "from a category cap. Keep estimates separate from posted transactions. Review the plan "
    "when income, household needs, or billing dates change, and preserve the date of each change."
)

DOCUMENTS = [
    ("BUD-101", "Monthly spending plan", "1.0", "2026-01-01", None, [
        "A monthly spending plan assigns take-home income to obligations, flexible needs, savings, and discretionary spending. Start with cleared income rather than a gross salary estimate. List rent, utilities, minimum debt payments, insurance, and support obligations before setting optional categories.",
        "Compare planned amounts with actual transactions at the end of each week. A variance is information, not a failure. If groceries exceed their plan, inspect timing, household size, and price changes before cutting another essential. Do not count a credit-card limit as available income.",
        "Budget code BUD-101 uses a calendar month by default. A household paid on a different cycle may map each pay period to the next set of obligations, but should keep one consistent reporting period. Carry-forward rules are separate and are defined in the dated policy documents.",
        "A useful review records the planned amount, actual amount, difference, and a short explanation. The user can then decide whether the next period should change. Avoid changing a target solely because of one unusual purchase or a pending refund that has not posted.",
        "For a simple example, a household with Rs 80,000 of monthly take-home pay might first record Rs 25,000 for rent and Rs 8,000 for groceries. The example is illustrative only; it is not a benchmark or recommendation for another household.",
        "When a plan is shared, identify who owns each entry and which account supplied it. Do not merge personal and household transactions without consent. Keep a separate note for amounts that are estimates, reimbursements, or one-time purchases.",
        "If a category repeatedly differs from plan, adjust the category or the plan assumptions rather than hiding the difference. Preserve the prior month so the user can see whether the adjustment improved clarity. A written explanation helps distinguish a structural change from a temporary exception.",
        "The coach can explain the method and retrieve the supporting guide. It cannot verify a bank balance, make a transfer, alter an account, or guarantee that a spending plan will prevent debt."
    ]),
    ("BUD-102", "Carry-forward policy", "2025.1", "2025-01-01", "2025-12-31", [
        "The 2025 budgeting guide used a month-end reset for flexible category caps. Unspent flexible funds did not automatically increase the next month's cap. A user could separately record a planned savings transfer, but the guide did not treat that plan as a completed bank transaction.",
        "This archived version is retained so questions about a 2025 statement can be answered in historical context. It is not the active rule for dates in 2026. A report should identify the month and policy version before comparing planned and actual amounts.",
        "Under code POL-CF-2025, a category could be manually rebalanced before the next month began. The user was expected to retain the original limit and the revised limit with an explanation. The archive did not support automatic carry-forward of a grocery or dining surplus.",
        "If a user asks about current carry-forward, do not quote this archived rule as current. Retrieve the current version and cite its ID. If the date is unknown, ask which period the user means rather than silently selecting the older policy.",
        "A planned transfer remained a plan until it appeared in an account record. Pending card transactions and expected reimbursements were not treated as cleared funds. This distinction prevents a spreadsheet estimate from being mistaken for a confirmed balance.",
        "The 2025 note did not authorize the coach to initiate transfers or provide tax advice. Users were responsible for checking account terms, due dates, and any fees with their financial institution.",
        "Example: a Rs 900 grocery surplus at the end of October 2025 stayed outside the November cap unless the user deliberately edited the new plan. The example only describes the archived guide and should not be applied to 2026.",
        "When reviewing old records, preserve the original date and version. Do not rewrite an old plan to look as if the newer policy was in force at the time."
    ]),
    ("BUD-102", "Carry-forward policy", "2026.1", "2026-01-01", None, [
        "The active 2026 budgeting guide permits an explicitly recorded unspent flexible-category amount to be carried into the next month once. The original cap and carried amount must remain visible as separate values so the next period is auditable.",
        "This version replaced POL-CF-2025 on 1 January 2026. The change applies only to newly planned periods from that date. It does not retroactively alter historical reports or turn an expected refund into cleared cash.",
        "Code POL-CF-2026 allows a user to move a documented surplus to one named category or to a separate savings goal. The user must choose the destination. The coach may explain options but must not choose a destination or execute a transfer.",
        "A carried amount is limited to one consecutive month. After that month, the user should either spend it, move it to a savings goal, or return it to the unallocated balance. The guide does not compound carry-forward across multiple months.",
        "Example: a recorded Rs 700 grocery surplus from April may be shown as a separate May carry-in. The May grocery cap remains visible, and the carry-in is not added to income. If May ends with a balance, it cannot be carried again under this version.",
        "A user can opt out and reset a category to its ordinary cap. Record the decision, period, and source amount. If a number is disputed, compare against posted transactions rather than using a summary total as proof.",
        "For dates before 2026, retrieve the archived 2025 policy. For dates on or after 2026-01-01, retrieve this version. The assistant must state the policy version when the distinction affects the answer.",
        "This policy is a planning convention, not a bank feature. It cannot change a card limit, account balance, loan payment, or merchant charge."
    ]),
    ("BUD-103", "Emergency reserve planning", "1.0", "2026-01-01", None, [
        "An emergency reserve is a planning label for money the household chooses to keep available for unexpected essential costs. The guide does not set a universal target because employment, dependants, location, insurance, and access to funds differ.",
        "A user may define a goal amount and date, then review progress against confirmed deposits. Keep the target, actual balance, and expected contribution distinct. A scheduled or intended contribution is not a completed deposit.",
        "Code GOAL-EM-01 is the example identifier for a reserve goal. It is not a financial product or account. The assistant can record the user's chosen target after explicit confirmation, but cannot open an account or move money.",
        "The plan can split an amount into manageable periods, provided the user chooses the amount and schedule. If the goal competes with a due bill or a high-cost obligation, show the trade-off and invite the user to verify terms before making a decision.",
        "An emergency may require immediate help that a budgeting guide cannot provide. For urgent safety, medical, or legal needs, contact appropriate local services or a qualified professional. Do not delay urgent assistance to complete a budget worksheet.",
        "Review access conditions, withdrawal limits, and fees directly with the financial institution. Do not label an investment or locked deposit as readily available unless its terms support that description.",
        "A reserve review should note the review date, the user's chosen target, and whether the number is an estimate or a verified account figure. Avoid exposing account credentials, full card numbers, or one-time passwords in a chat.",
        "The guide is descriptive rather than prescriptive. It helps organize a user's stated plan and does not guarantee that any particular amount is sufficient."
    ]),
    ("BUD-104", "Debt payment organizer", "1.0", "2026-01-01", None, [
        "A debt organizer records each obligation, due date, minimum payment, stated rate, and the source of the information. Verify current balances and terms with the lender. A copied figure may be stale, and a minimum payment does not necessarily reduce principal quickly.",
        "The snowball method orders debts by balance, while an interest-rate-first method orders them by stated rate. These are comparison frameworks, not automatic recommendations. Fees, promotional periods, penalties, and cash-flow needs can change the practical comparison.",
        "Identifier DEBT-AV-02 labels the example interest-rate-first worksheet. Identifier DEBT-SN-01 labels the smallest-balance worksheet. Neither is a lender account ID. The user selects a method after reviewing verified terms.",
        "Always keep required payments visible before modeling an optional extra payment. Do not recommend skipping an essential bill or taking a new loan based only on the coach's summary. If a payment cannot be made, the user should contact the creditor or a qualified adviser.",
        "A plan should distinguish posted payments from scheduled payments. Record the amount, date, and confirmation source. A pending transfer may fail or arrive late; do not report it as settled until the account shows completion.",
        "When comparing two methods, explain the assumptions and show the order of debts without claiming certainty about future interest. Rates may be variable and fees may apply. Ask the user to confirm the exact rate and balance date.",
        "Do not collect full account numbers, login details, card security codes, or authentication codes. A short label such as 'card A' is enough for a private worksheet.",
        "The tool stores only user-provided notes and a local goal record. It does not connect to a lender, submit a payment, negotiate terms, or provide legal advice."
    ]),
    ("BUD-105", "Cash-flow calendar", "1.0", "2026-01-01", None, [
        "A cash-flow calendar places expected income and due dates on a timeline. It should mark whether each entry is confirmed, estimated, or recurring. A monthly total can look adequate while a bill falls before the next pay date.",
        "Use the date a payment is due, not only the date a statement is issued. Include a reminder buffer chosen by the user. Do not assume that a weekend or bank holiday will move a due date in a particular way; verify the provider's terms.",
        "Code CAL-28 is a sample reminder label for a payment scheduled near the 28th. It is a local planning tag, not a bank routing code. Calendar entries should not contain passwords or complete payment credentials.",
        "For irregular pay, enter only the amount and date supported by a reliable estimate. Keep a conservative scenario separate from a typical scenario. If income changes, update the plan before treating an expected amount as available.",
        "A calendar can show rent, utilities, minimum debt payments, and savings contributions without authorizing any action. The database write in this demo creates a goal record only after confirmation; it never schedules a bank transaction.",
        "A user who says 'and for Saturday?' may be referring to a date or category already present in the conversation. Resolve that reference from the active state; if two entries could match, ask which one they mean.",
        "At a weekly review, mark cleared items, leave pending items visibly pending, and compare the next seven days with the available cash information the user supplies. Avoid presenting a forecast as a verified balance.",
        "Calendar estimates should be reviewed after a missed or shifted payment. Preserve the original value and the correction date so the household can understand what changed."
    ]),
    ("BUD-106", "Sinking funds", "1.0", "2026-01-01", None, [
        "A sinking fund is a user-defined label for planning a known future expense by setting aside amounts over time. Examples include annual insurance, school costs, repairs, or travel. The label alone does not create a separate bank account.",
        "Record the expected expense, target date, chosen target, and any amount already set aside. Divide the remaining target across the user's chosen number of pay periods only after confirming the dates and assumptions.",
        "Identifier SF-ANNUAL-01 is a sample annual-expense worksheet. It is not an account number. Keep the goal record separate from the household's available balance unless that balance has been verified.",
        "If the date or amount changes, preserve the earlier target as superseded and create a dated replacement. The current view should show only the active target, while the history remains available for an audit unless the user requests deletion.",
        "A sinking fund is different from emergency savings: the expense is anticipated, while an emergency reserve covers unexpected essential needs. A user can choose different labels or combine them, but the assistant should not silently merge the records.",
        "If the user asks how much to contribute, calculate only from the provided target and time periods and state the arithmetic. Do not imply that the calculated contribution is affordable without current income and obligations.",
        "For example, a chosen Rs 12,000 annual expense over six remaining monthly contributions yields Rs 2,000 per month before any prior set-aside. This is arithmetic, not a recommendation to sacrifice other obligations.",
        "Review the plan when an expense is paid, postponed, or cancelled. Mark the event accurately and avoid retaining a cancelled target as if it were still due."
    ]),
    ("BUD-107", "Recurring bills and utilities", "1.0", "2026-01-01", None, [
        "A recurring-bill worksheet tracks a provider label, typical amount, due date, frequency, and the date the amount was last checked. Bills can vary by use, tariff, tax, or provider change, so the latest statement is stronger evidence than an old average.",
        "Code BILL-UTIL-07 is a sample category for a utility bill. It is not a provider identifier. Enter a range if the amount varies and explain the period used to calculate it.",
        "Separate essential bills from optional recurring services for review, but do not assume that cancelling a service has no fee or notice period. Check the provider's current terms before taking action.",
        "A reminder is not a payment instruction. The budgeting assistant can list a due date and read a user-provided record, but it does not initiate a debit, access an account, or guarantee that a payment was made.",
        "When a household shares an obligation, record who is responsible and whether a reimbursement is expected. Keep an expected reimbursement separate from cleared cash and avoid double-counting it as both income and a reduction in expense.",
        "If a bill changes unexpectedly, compare the current statement with the prior period and identify the difference. Do not invent a provider explanation. The user may need to contact the utility provider directly.",
        "Review recurring costs periodically and note whether a service is still used. The worksheet can show a decision to review, but the user decides whether to change a service after checking any contract or penalty.",
        "Keep personal identifiers minimal. A provider nickname and the final few digits of a reference may be enough for private notes; never store authentication codes or full payment card data."
    ]),
    ("BUD-108", "Savings goal workflow", "1.0", "2026-01-01", None, [
        "A goal record contains a purpose, target amount, due date, and creation time. The assistant must ask for any missing slot, read the completed values back, and obtain an explicit confirmation before inserting the record.",
        "The example code GOAL-NEW-01 identifies a newly created goal in the sample workflow. The SQLite database returns its own numeric row ID. A successful insert is the only evidence that the goal was saved.",
        "A user can revise a target by stating the new amount or date. Keep the active target distinct from earlier versions, and ask whether the user intends to update the existing goal or create a separate one if that is ambiguous.",
        "Do not treat 'maybe', a question, or a prior unrelated yes as consent. If the user cancels, discard pending slots. If a confirmation is unclear, ask again without calling the write tool.",
        "A date must use an unambiguous ISO format in this demo. If a user supplies a relative phrase such as 'next month', ask for the exact date or confirm the interpretation before saving.",
        "Goal progress requires a verified contribution amount. This demo does not connect to a bank, so it cannot update progress from transactions. It can only preserve the user's chosen target and due date.",
        "Before creating a goal, explain that an entry is a planning record, not a transfer or a financial product. Do not promise that reaching the target will guarantee a particular outcome.",
        "If a goal contains an amount in another currency, clarify the currency before recording it. This variant's sample database stores INR values and does not perform currency conversion."
    ]),
    ("BUD-109", "Subscription review", "1.0", "2026-01-01", None, [
        "A subscription review records the service label, billing frequency, last known charge, renewal date, and whether the user recognizes the charge. A recurring-looking card entry may still be a one-time transaction or an unfamiliar merchant.",
        "Code SUB-CHK-09 is the checklist identifier for reviewing a recurring charge. It does not identify a real provider. Verify the statement and merchant terms before concluding that a charge is a subscription.",
        "Compare monthly and annual billing using the actual total, renewal timing, cancellation notice, taxes, and refund rules. Do not describe a discounted annual price as savings if the user may not use the service for the full year.",
        "If a charge is unauthorized, the user should contact the financial institution through its official channel and follow the institution's dispute process. The assistant cannot reverse a payment or determine whether fraud occurred.",
        "Keep a cancellation reminder separate from a cancellation confirmation. A user should retain the provider's confirmation and check a later statement for any final charge.",
        "When reviewing a shared subscription, clarify who uses it and who pays. Do not infer consent to cancel a service that another household member relies on.",
        "A budget estimate may use the latest known monthly cost, but it must be labelled with the statement date. If the price changes, replace the active amount and preserve the older value as history unless the user asks to forget it.",
        "The guide is intended for organization and comparison. It is not an endorsement of any service and does not provide legal conclusions about a provider's cancellation terms."
    ]),
    ("BUD-110", "Privacy and safe budgeting notes", "1.0", "2026-01-01", None, [
        "Store the minimum information needed for a budgeting conversation. A category, amount, and month are usually enough. Do not enter passwords, PINs, one-time codes, complete card numbers, account credentials, or government identity numbers.",
        "The local demo separates records by user ID and reads only that user's memory. A production deployment must enforce authentication and authorization at every data access; a user label typed into a command line is not identity verification.",
        "Code PRIV-01 marks a privacy reminder in the sample knowledge base. It is not a security certification. Users should review where local files are stored and remove sensitive data from a device they do not control.",
        "A forget request should remove the selected semantic fact, its vector representation, and any episode or reflection that repeats it. If the key is ambiguous, ask which information should be removed instead of deleting broadly.",
        "Logs are required for reproducible model and tool measurements, but they should not contain passwords or unnecessary financial identifiers. This project logs call metadata and keeps user utterances in the scripted dataset, not in every call-log row.",
        "When sharing evaluation files, inspect them for personal values and identifiers. Synthetic examples should be clearly identified, and real account statements should not be copied into a public repository.",
        "Use a separate user identifier for each authorized participant. Do not answer a question about another person's financial preferences from one user's memory. A privacy boundary is more important than a seemingly helpful guess.",
        "Data retention and deletion obligations depend on the deployment and applicable policy. This educational sample is not a complete compliance or security solution."
    ]),
    ("BUD-111", "Irregular income worksheet", "1.0", "2026-01-01", None, [
        "An irregular-income worksheet stores a range of dates and amounts, the source of each estimate, and whether the amount has cleared. Separate guaranteed income from commissions, freelance invoices, gifts, and reimbursements.",
        "Code INC-RNG-11 labels a sample range-based scenario. A low, typical, and high estimate can help a user see uncertainty, but the assistant should not present a forecast as guaranteed income.",
        "Plan fixed obligations against the income the user considers reliable. Treat uncertain receipts separately until confirmed. The user chooses how to handle a shortfall; the assistant can identify which due dates occur before the next expected receipt.",
        "When a payment is delayed, update its status and date rather than deleting the estimate. Preserve the source and time of the correction so later analysis does not assume the original forecast was accurate.",
        "A monthly average can hide a timing gap. A calendar view may be more useful when due dates cluster before income arrives. Ask for the relevant dates if a question depends on a specific pay period.",
        "Do not recommend borrowing against an expected invoice or using a credit product based only on this worksheet. The user should verify fees, terms, and alternatives with a qualified source.",
        "A budget can use a conservative scenario chosen by the user and retain a separate optimistic scenario for planning. Clearly label each assumption. Do not silently switch between them when answering a follow-up.",
        "This guide helps structure information; it cannot verify an employer, client, payment, or bank balance."
    ]),
    ("BUD-112", "Category definitions", "1.0", "2026-01-01", None, [
        "Categories describe how a household chooses to group transactions. Common groups include housing, food, transport, health, education, communication, debt obligations, savings, and discretionary spending. The same merchant may belong to different categories depending on the purchase.",
        "Code CAT-FOOD-12 is a sample groceries label; CAT-DINE-13 is a sample dining label. Codes are local tags, not standardized bank categories. Keep them stable within a report so comparisons remain understandable.",
        "Do not force a mixed transaction into a single category when the receipt can reasonably be split. Record the method and keep the original transaction total so the split can be checked.",
        "A category cap is a planning limit selected by the user. It is not a prediction, a guarantee, or an instruction to avoid necessary care. If an essential cost exceeds the cap, record the actual amount and review the plan.",
        "Transfers between the user's own accounts should not be counted as both income and spending. A transfer may be a movement of funds rather than a new source or use. Confirm the transaction type before classifying it.",
        "Use an uncategorized label temporarily when the transaction is unclear. Ask for a merchant or purpose only if the user is comfortable sharing it. Do not infer sensitive details from a merchant name.",
        "When category names change, map old and new names in a dated note. Preserve the original records and disclose any reclassification in a comparison.",
        "A category framework is personal. Two households may use different labels and still describe the same underlying cash flow."
    ]),
    ("BUD-113", "Credit and payment terminology", "1.0", "2026-01-01", None, [
        "A credit limit is the maximum amount a lender may allow under its terms, not money owned by the user. Available credit is not income. Keep an outstanding balance, statement balance, minimum payment, and limit as separate values.",
        "Code CRD-UTIL-13 labels a utilization worksheet example. Any percentage calculation must identify the balance and limit dates used. The result is a snapshot, not a score prediction or a guarantee of how a lender will report.",
        "A billing statement may show a statement date and a separate due date. The user should verify both with the issuer. A payment scheduled before the deadline is not necessarily posted before the deadline.",
        "Fees, interest, grace periods, and reporting practices depend on the card agreement and local rules. Do not invent a universal threshold or advise a user to carry a balance to build credit.",
        "If a user asks about a disputed transaction, explain that the issuer's official dispute channel is the source for deadlines and required evidence. The assistant cannot decide whether a charge is valid or guarantee a dispute result.",
        "Never ask for a full card number, CVV, PIN, password, or one-time code. A short private nickname can distinguish records without collecting authentication data.",
        "When entering a debt payment into a plan, distinguish required minimums from extra amounts the user chooses. Recheck the latest statement before calculating a comparison.",
        "This educational note does not replace the card agreement or independent financial advice."
    ]),
    ("BUD-114", "Unexpected expense and shortfall", "1.0", "2026-01-01", None, [
        "When an unexpected expense occurs, first record its amount, date, and whether it is paid, pending, or estimated. A shortfall is a change in circumstances, not evidence that the user failed to follow a plan.",
        "Code ERR-402 labels a missing or unverified amount in this demo. It is not a bank error code. If a tool lacks current data, the assistant should say what is missing and ask for a safe, non-sensitive value.",
        "Review upcoming essential bills and dates using user-provided information. Do not tell a user to skip rent, medicine, or a required payment. If they cannot meet an obligation, suggest contacting the relevant provider or a qualified local adviser.",
        "A response should distinguish options from instructions. The user may decide to revise a flexible category, postpone a discretionary purchase, or seek support. The assistant should not choose a loan, investment, or payment product.",
        "If the user reports an urgent safety or health concern, prioritize appropriate emergency resources over budget optimization. A budget workflow is not an emergency service.",
        "Update an estimate only after the user states the new amount or supplies a verified record. Keep a dated note about why a plan changed and avoid presenting the prior estimate as current.",
        "Use a brief clarification when the amount, currency, or due date is ambiguous. If the user gives conflicting values, repeat the alternatives and ask which is correct.",
        "The sample coach cannot access accounts, negotiate, transfer funds, or guarantee a creditor's response."
    ]),
    ("BUD-115", "Budget data glossary", "1.0", "2026-01-01", None, [
        "Take-home income means the amount the user says is available after deductions; the coach does not verify payroll. Planned spending is a target. Actual spending is a transaction amount confirmed from a record. An estimate is neither.",
        "Code DEF-115 is a glossary record ID. A source turn records where a conversational fact came from. A superseded fact remains historical but must not be used as the current value. A forgotten fact must not be retrieved.",
        "A recurring expense repeats on a schedule, while an irregular expense occurs occasionally or varies substantially. An obligation is a payment the user considers required, but the assistant should not infer a legal priority from the label alone.",
        "A budget period is the time span used for comparisons. The default here is a calendar month, but a user may describe a different pay cycle. State the period in a calculation to avoid comparing incompatible totals.",
        "A due date is the date the provider says payment is required. A transaction date is when an event was recorded. A posting date is when the institution processed it. These dates may differ.",
        "A planned savings goal is a record of intention. A deposit is a transaction. The database in this assignment creates only the former and does not transfer money.",
        "A retrieval citation identifies the exact document chunk supplied to the model. A citation proves which guide text was retrieved; it does not prove that the guide applies to every person's situation.",
        "When a term is unclear, ask what the user means rather than silently assigning a definition."
    ]),
    ("BUD-116", "Worked budgeting examples", "1.0", "2026-01-01", None, [
        "Example one: a user states monthly take-home income of Rs 85,000 and rent of Rs 26,000. The remaining arithmetic is Rs 59,000 before groceries, utilities, debt, savings, transport, and other costs. This is not a recommendation.",
        "Example two: a grocery cap is Rs 9,000 and the recorded month-to-date total is Rs 6,400. The simple remaining planned amount is Rs 2,600, provided both numbers cover the same month and the transactions are current.",
        "Example three: a user creates a Rs 20,000 emergency-fund goal due on 31 January 2027. The assistant asks for purpose, amount, and date, repeats them, and waits for explicit confirmation before saving the planning record.",
        "Example four: a user changes monthly income from Rs 85,000 to Rs 92,000. The old fact becomes superseded. A current answer should use Rs 92,000 and should identify the update turn if the user asks when it changed.",
        "Example five: a user asks for an old rent amount to be forgotten. The semantic fact and related episodes or reflections are removed. A later question about that forgotten amount receives an honest 'not stored' response.",
        "The examples use fictional figures and IDs. They demonstrate arithmetic and interaction mechanics only. They must not be copied into a user's budget as assumed values.",
        "In a code-mixed question, query rewriting can map a known Hindi term such as 'kiraya' to 'rent' before retrieval. Preserve the user's original meaning and show the retrieved citation rather than translating a policy into a new rule.",
        "If two examples or policies conflict, check the document version and validity date. Ask which period the user means when the date determines the correct answer."
    ]),
    ("BUD-117", "Local tool and error codes", "1.0", "2026-01-01", None, [
        "Tool identifiers are local labels for explaining this educational system. TOOL-KB-01 refers to knowledge search, TOOL-DB-02 refers to a profile read, and TOOL-GOAL-03 refers to a confirmed goal insert.",
        "ERR-404 means a requested document or index file was not found in the local demo. ERR-422 means a required slot or input value is invalid. ERR-503 means an optional local model service is unavailable. These labels are not external financial institution codes.",
        "A failed search must not be reported as a successful answer. A missing index should be rebuilt with the documented command. A missing model should be reported explicitly rather than replaced with a silent default.",
        "Database writes use parameterized SQL and return a row ID on success. If validation fails, no insert should occur. A write must happen only after a clear confirmation tied to the currently displayed values.",
        "Every model, embedding, or tool call appends an actual runtime record to logs/calls.jsonl. A failed run remains in the log. Start a new run ID for a retry; do not edit or remove prior records.",
        "The log contains timing and token counts, not a transcript. Tool results should contain only the metadata needed to reproduce a measurement. Keep secrets and personal account identifiers out of the log.",
        "For evaluation, calculate tables from the run log and probe results. Do not type a desired metric into a CSV. When a run is incomplete, report the failure and the command that produced it.",
        "These codes describe the sample implementation and may change between software versions. Keep the code-to-meaning mapping beside the versioned source."
    ]),
]


def main() -> None:
    directory = Paths().kb
    directory.mkdir(parents=True, exist_ok=True)
    for doc_id, section, version, valid_from, valid_to, paragraphs in DOCUMENTS:
        filename = f"{doc_id.lower()}-v{version.replace('.', '-')}.md"
        header = (
            "---\n"
            f"doc_id: {doc_id}\n"
            f"section: {section}\n"
            f"version: {version}\n"
            f"valid_from: {valid_from}\n"
            f"valid_to: {valid_to if valid_to else 'null'}\n"
            "---\n\n"
            f"# {section} (v{version})\n\n"
        )
        text = header + "\n\n".join(paragraphs + [COMMON])
        (directory / filename).write_text(text + "\n", encoding="utf-8")
    total_words = sum(
        len(path.read_text(encoding="utf-8").split())
        for path in directory.glob("*.md")
    )
    print(f"Wrote {len(DOCUMENTS)} versioned documents; total word count={total_words}.")
    if not 4000 <= total_words <= 8000:
        raise SystemExit("Knowledge base must contain 4,000-8,000 words.")


if __name__ == "__main__":
    main()
