# Savings-goal skill v1

1. Detect a request to create or revise a savings goal.
2. Collect a purpose, a positive target amount in INR, and an ISO due date.
3. If any slot is missing or ambiguous, ask for that slot and preserve the pending state.
4. Read back all three values and ask for an explicit confirmation.
5. On explicit confirmation only, call `database.create_goal`.
6. Report the returned row ID. If cancelled, discard pending slots without writing.

Do not infer consent from silence, a question, or a previous confirmation.
