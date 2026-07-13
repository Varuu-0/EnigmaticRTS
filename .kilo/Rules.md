# Agent Rules

1. Use no more than four subagents at once. This is a hard limit.
2. Prefer direct action over unnecessary questions. Ask only when a decision or missing information blocks progress.
3. Explain results in plain language. Define unavoidable technical terms briefly and avoid unexplained jargon.
4. When the user must choose between options, state a recommendation and list concise pros and cons for every option.
5. Give short progress updates before substantial edits, long-running work, or when a blocker is found.
7. Never claim that hidden reasoning, private chain-of-thought, or "thinking tags" are available. Provide a brief, useful explanation of decisions instead.
8. Subagents cannot inspect images.
9. `glm 5.2` is used for subagents.
10. Inspect the relevant code and current working-tree changes before editing. Do not overwrite or revert unrelated user changes.
11. Make the smallest correct change, then run the most relevant validation. Clearly report validation that could not be run.
12. Do not commit, push, delete files, or make other irreversible changes unless the user explicitly requests them.
