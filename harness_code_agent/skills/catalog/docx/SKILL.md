---
name: docx
description: Read, inspect, create, edit, convert, and verify Microsoft Word .docx files. Use when a task includes a .docx path or a Word attachment.
---

# DOCX

Treat the attachment's `local_path` as the source of truth. The attachment layer deliberately does not extract or embed the document body in the model request.

## Workflow

1. Confirm the exact source path, desired result, and output format. Preserve the original unless the user explicitly requests an in-place edit.
2. Inspect the package before editing. Use `python-docx` for paragraphs, headings, tables, sections, headers, footers, styles, relationships, and core properties. Remember that text may also live in tables, text boxes, comments, footnotes, headers, or tracked revisions.
3. Choose the smallest suitable route:
   - Read or summarize ordinary content: extract paragraphs and table cells in document order with `python-docx`.
   - Create or structurally edit a DOCX: use `python-docx`, reusing the source document's styles where possible.
   - Tracked changes, comments, fields, text boxes, or unsupported WordprocessingML: inspect and edit the underlying XML only for the specific structure required.
   - Layout-sensitive verification or PDF delivery: convert a copy with LibreOffice, then render and inspect the PDF.
4. Use `run_bash` for read-only probes. For non-trivial edits, create a short, task-specific Python script in the workspace. Do not install packages or invoke network services without user authorization.
5. Save to a clearly named output, reopen it with `python-docx`, and verify the requested paragraphs, tables, styles, relationships, and section settings.
6. Report the exact output path and any fidelity limits, especially unsupported Word features or conversion differences.

## Safety and quality

- Quote paths and pass them as arguments; never interpolate document text into shell commands.
- Do not expose ZIP bytes, document XML, or Base64 in prompts, logs, traces, or summaries unless a tiny XML excerpt is necessary to diagnose a specific issue.
- Preserve styles, numbering, hyperlinks, images, headers, footers, section breaks, and page settings unless the task asks to change them.
- Do not rebuild an existing document from plain extracted text when a targeted edit can preserve its structure.
- `python-docx` does not render Word layout. For pagination, wrapping, fonts, fields, and image placement, conversion plus visual inspection is required.
- If a dependency or converter is unavailable, state the missing capability and a concrete install option. Do not claim visual verification.

Load `catalog/docx/REFERENCE.md` with `read_skill_file` only when you need structure coverage, XML guidance, conversion, or validation details.
