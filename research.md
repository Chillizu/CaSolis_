# Research: Review of PLAN_BRAIN_HAND_v2.md

## Summary
The v2 revision fixes the most immediate technical blocker (46M trained on 2k samples) by switching to a frozen Sentence Transformer backbone with a tiny classification head, and it correctly reframes Qwen away from bash generation toward parameter reasoning. However, the **biggest remaining problem is still the lack of a real online learning loop**: the agent cannot learn new intents, adapt its policy from outcomes, or grow beyond the 10 hand-coded action types. RND and Qwen are incremental improvements, but they do not turn the system into an autonomous learner.

## Findings
1. **Parameter/data ratio issue is fixed.** Replacing the 46M-from-scratch model with `all-MiniLM-L6-v2` (22M frozen) plus a `Linear(384, 10)` head (~4K trainable params) makes 2,000 samples a reasonable training set. This is the right call and directly addresses the original overfitting concern. [Source](PLAN_BRAIN_HAND_v2.md)
2. **Qwen's role is improved but still heavier than necessary.** Moving Qwen from "bash translator" to "parameter reasoner" reduces hallucination surface because deterministic templates generate the final command. However, using a 1.5B model to emit `pattern=root` or `path=/etc/passwd` is still overkill; many parameters could be deterministic (regex, path completion, or simple heuristics). PIPE and HELP intents also push ambiguous, open-ended reasoning back onto Qwen, reintroducing fragility. [Source](PLAN_BRAIN_HAND_v2.md)
3. **RND on state transitions is conceptually the right fix for Noisy TV.** Applying curiosity to changes in a structured state (visited paths, current directory, learned intents) rather than to raw command output avoids the `date`/`ps`/`uptime` trap. A caveat: if "recent N steps" includes the stochastic output of `date`, the state itself becomes noisy and the Noisy TV problem returns; the state representation must exclude ephemeral output strings. [Source](PLAN_BRAIN_HAND_v2.md)
4. **10 intents are better than 20, but still a hand-coded action-type ontology.** The author acknowledges this and recasts intents as "Action Types," which is honest. Still, EXPLORE and HELP are underspecified, and the agent has no mechanism to invent or specialize new intents. The system remains a classifier over a human-designed menu, not an open-ended thinker. [Source](PLAN_BRAIN_HAND_v2.md)
5. **No online learning mechanism is the largest unresolved gap.** The RND predictor trains online, but the main policy (intent classifier) is trained once offline on 2k labeled samples. There is no reward/utility model, no outcome-driven policy update, and no way for the agent to discover that some intent sequences work better than others. Curiosity alone produces exploration signals with no guarantee of competence improvement. [Source](PLAN_BRAIN_HAND_v2.md)
6. **The 24h MVP recommendation from the previous review has been partially accepted, but v2 still adds under-specified complexity.** RND (200K predictor) and Qwen 1.5B both introduce latency, failure modes, and infrastructure burden before the core loop (state → intent → template → execute → observe) is proven to work end-to-end.

## Sources
- Kept: PLAN_BRAIN_HAND_v2.md (project file) — the revised architecture plan under evaluation.
- Kept: Response/reply_to_kimi.md (project file) — maps the author's claimed fixes back to the original five criticisms.

## Gaps
- No empirical evaluation plan: what tasks, what sandbox, what success metric?
- No definition of how Qwen outputs are validated before being passed to templates (JSON schema? retries? fallback?).
- No safety/allowlist details beyond a brief mention of "whitelist verification + path security check."
- No data or training procedure for the RND predictor; its ~200K parameters have no stated dataset.

## Supervisor coordination
None needed; this is a document-based technical review with no blockers requiring a decision.
