# Emergent Properties — AI Agent Playbook

This guide keeps every AI contributor aligned with the project's creative goals, tooling expectations, and collaboration rituals with the human director.

## 1. Grounding & Inputs
- Always read `PLAN.md` and `DESIGN.md` before picking up work; treat them as the single source of truth for scope and aesthetics.
- When in doubt, ask the human for clarifications instead of guessing—especially around visual/experiential intent.
- Prefer incremental pull-style updates so the human can review and steer early.

## 2. Workflow Expectations
- Work on one well-defined task at a time; if a task feels vague, write down assumptions in your worklog before coding.
- Keep diffs tight: scaffold, test, and document as you go rather than batching large unreviewable changes.
- Leave lightweight comments only where complex logic would otherwise slow the next agent down.

## 3. PLAN.md Maintenance
- Treat `PLAN.md` as a living contract. Whenever you finish (or materially advance) a checklist item, update the relevant box (e.g., switch `- [ ]` to `- [x]`).
- If you adjust scope, add or reorder tasks directly in `PLAN.md` so future agents inherit the updated roadmap.
- Do not remove work logged by the human; instead, annotate with clarifications or follow-up TODOs.

## 4. DEVLOG.md Maintenance
- Every commit needs to include a DEVLOG entry.
- Only append to file.
- Add date to every entry.

## 5. Testing & Quality Gates
- For every new feature or bugfix, add or update a Jest test whenever it is technically feasible. Place tests beside the source (e.g., `feature.test.js`) or under a `__tests__` directory consistent with the surrounding code.
- Keep tests focused on observable behavior, not implementation details. Mock browser APIs or Web Audio nodes as needed.
- Run the relevant Jest suites (`npm test`, `npx jest <pattern>`, etc.) before handing updates back; document failures you cannot fix.

## 6. Implementation Notes
- Mirror the directory structure outlined in `DESIGN.md` (e.g., `index.html`, `styles/`, `js/` modules) and respect existing naming conventions.
- Favor pure modules and explicit data flow between systems like `Renderer`, `AudioManager`, `FeatureExtractor`, and `ParticleField`.
- Maintain lint-friendly, ES module–compliant code; default to modern syntax (async/await, const/let, template literals).

## 7. Handoff Checklist
- Summarize what changed, how to verify it (tests, manual steps), and any open questions.
- Mention which PLAN items you touched and confirm their status boxes match reality.
- Highlight new risks, TODOs, or assumptions so the next agent or human can act without spelunking through diffs.

Stay deliberate, keep the vibe intact, and document just enough for the next agent to pick up momentum instantly.
