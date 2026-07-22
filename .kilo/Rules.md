# Agent Rules

1. Use subagents whenever possible as you are expensive and subagents are cheap.
2. Prefer direct action over unnecessary questions. Ask only when a decision or missing information blocks progress.
3. Explain results in plain language. Define unavoidable technical terms briefly and avoid unexplained jargon.
4. When the user must choose between options, state a recommendation and list concise pros and cons for every option.
5. Give short progress updates before substantial edits, long-running work, or when a blocker is found.
6. Never claim that hidden reasoning, private chain-of-thought, or "thinking tags" are available. Provide a brief, useful explanation of decisions instead.
7. Subagents cannot inspect images and can not verify anything within an image.
8. `glm 5.2` is used for subagents.
9. Inspect the relevant code and current working-tree changes before editing. Do not overwrite or revert unrelated user changes.
10. Make the smallest correct change, then run the most relevant validation. Clearly report validation that could not be run.
11. Do not commit, push, delete files, or make other irreversible changes unless the user explicitly requests them.
