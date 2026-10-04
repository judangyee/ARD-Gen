## Ponytail rules

Default to the laziest solution that actually works — not careless, efficient.

- Ladder before writing code: does this need to exist at all (YAGNI)? → reuse what's already in the codebase → stdlib → native platform feature → already-installed dependency → one line → only then, minimal new code.
- Bug fixes: root cause in the shared function, not a patch in every caller.
- No unrequested abstractions, boilerplate, or "for later" scaffolding.
- Shortest working diff. Code first; explanation in at most three short lines (`[code] → skipped: [X], add when [Y].`).
- Mark deliberate shortcuts with a `ponytail:` comment naming the ceiling and upgrade path.
- Never simplify away: input validation at trust boundaries, error handling, security, accessibility, or anything explicitly requested.
- Never skip understanding the problem — trace the real flow before picking a rung.
- Non-trivial logic (branch, loop, parser, money/security path) leaves one small runnable check (assert/`__main__`/`test_*.py`).
- Levels: `lite` / `full` (default) / `ultra`. Toggle with "ponytail" / "stop ponytail".
