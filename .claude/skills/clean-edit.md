# Skill: Clean Edit & Summary
A skill to apply code changes cleanly without cluttering the chat with code replacement snippets.

## When to use
Use this skill whenever the user asks to modify, refactor, or optimize existing code and you need to call file editing tools.

## Instructions
1. Plan the code changes mentally or in a brief scratchpad.
2. Execute the necessary tool calls (`edit_file`, etc.) to modify the code.
3. In your final response to the user, **CRITICAL**: Do not print the code snippets or diff blocks that you just replaced.
4. Provide only a 2-3 line bulleted summary explaining *what* changed and *why*.