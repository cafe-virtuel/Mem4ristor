# EDISON Recommendations for Future Hermes Sessions

## Context
EDISON = Evidence-Driven Investigation Systematic Output Notification.
Session 2026-05-31: EDISON caught a critical inconsistency in the preprint.
The contradiction was invisible from inside the manuscript.

## Core Recommendations

1. **Self-verify numbers against the PRIMARY column in metrics tables before propagating them anywhere.**
   Example: When reading H_stable from Table_metrics, verify the value exists in the synchronie column before using it in the abstract.

2. **When EDISON flags a discrepancy, stop and resolve it before moving forward.**
   EDISON catches what self-review misses. The correction cost was 6 hours of re-analysis, but the alternative was publishing false results.

3. **Keep synchronie (column synchronisation) as the primary ranking, not H_cont.**
   H_cont is a secondary indicator. The hierarchy is: synchronie = PRIMARY, H_cont = SECONDARY. Never reverse this without justification.

4. **If you read a value from a file, verify it exists before using it.**
   This seems obvious but it's the root cause of the preprint incident. AUDIT-011 (Claude) had the same failure mode.

5. **Bad results stay documented as bad results.**
   Julien's epistemology: "la verite est notre seul juge." Never reword history to make bad results look better. A failed hypothesis is still valuable science.

## Secondary Recommendations

6. **Work sessions are LONG — write recommendations for the next you.**
   After sessions with 50+ tool calls, write recommendations for your next self. Save to: `D:/ANTIGRAVITY/TEST_HERMES/mem4ristor-v2-main/docs/papers/recommendations/EDISON_RECOMMENDATIONS.md`

7. **Latin characters only in Telegram and documents.**
   No Unicode escapes (\u00b1, \u00d7) which render as raw text on mobile. Write symbols as full words.

8. **Reprendre un projet de recherche: toujours dans l'ordre.**
   PROJECT_STATUS.md -> PROJECT_HISTORY.md -> WORK_LOG.md -> AUDIT_LOG.md. Never skip steps when resuming.

---

Last updated: 2026-05-31 (session AUDIT-020)