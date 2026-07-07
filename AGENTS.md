# AGENTS.md (PySide6 conventions)

## Python environment
- This project uses a conda environment named `learning_pyside6`.
- For all Python commands, use:
```powershell
conda run -n learning_pyside6 python```

## IMPORTANT list
- NEVER change or update any file ended with `_ui.py`

## Slots (MUST)
- Name every slot/signal-handler: `_on_<objectName>_<action>`
- `<objectName>` = widget.objectName() exactly.
- `<action>` = short snake_case verb for the signal.
  Examples: `_on_saveButton_clicked`, `_on_searchInput_text_changed`, `_on_modeCombo_index_changed`
- For every slot function declared, write a comment above it and indicate clearly that they are a slot function.

## Summaries (MUST)
- Put a comment block immediately ABOVE every `class` and `def` you create.
- For every `class`, describe the summary, what it does generally as descriptive as possible.
- For every `def`, describe the summary, what it does generally, and also the input and the return values.

## Logic comments (MUST; junior-friendly)
- Add comments explaining the WHY/intent for each logical block you generate or change.
- Always comment: signal connection, UI state changes, validation/transforms, non-obvious conditionals/loops, async/timers/threads
- Indicate every signal connection `.connect()` and slot function declaration 
- Avoid commenting trivial self-evident lines unless it clarifies WHY.
- Use simple words; assume a junior dev is reading.

## Qualisys / QTM (MUST)
- Use the official SDK `qtm-rt` (`import qtm_rt`) and follow `docs/qualisys/qtm_rt_api_notes.md`; do NOT invent method names or packet fields. :contentReference[oaicite:1]{index=1}
- Follow the example code `docs/qualisys/example_qtm_connect.py` for the starting point 
- Streaming is asyncio-based; never block the Qt UI thread, and assume packets/components can be missing—handle `None`/presence checks defensively. :contentReference[oaicite:2]{index=2}
