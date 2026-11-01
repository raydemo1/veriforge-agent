---
name: pdf
description: Read, inspect, extract, create, edit, merge, split, render, and verify PDF files. Use when a task includes a .pdf path or an application/pdf attachment.
---

# PDF

Treat the attachment's `local_path` as the source of truth. The attachment layer deliberately does not embed PDF bytes or extracted text in the model request.

## Workflow

1. Confirm the exact source path and requested outcome. Keep outputs inside the workspace unless the user explicitly authorizes another destination.
2. Inspect before changing anything: record page count, encryption state, metadata, and whether pages contain extractable text. Never infer that a PDF is searchable from its appearance alone.
3. Choose the smallest suitable route:
   - Searchable text or metadata: use `pypdf`.
   - Tables or layout-aware extraction: use `pdfplumber`.
   - Visual inspection, scanned pages, or layout verification: render selected pages with `pypdfium2`; use OCR only when extraction shows the page is image-based.
   - Merge, split, rotate, crop, or metadata edits: use `pypdf`.
   - New PDF generation: use ReportLab, then render and inspect the result.
4. Use `run_bash` for read-only probes. For a non-trivial transformation, create a short, task-specific Python script in the workspace rather than an unreadable one-liner. Do not install packages or invoke network services without user authorization.
5. Preserve the original file by default. Write to a clearly named output path, then reopen it and validate page count, extractability, and the requested change.
6. Report the exact output path and any limits: failed OCR, unsupported encryption, missing fonts, damaged objects, or layout differences.

## Safety and quality

- Quote paths and pass them as arguments; do not interpolate document-derived text into shell commands.
- Do not expose raw binary or Base64 in prompts, logs, traces, or summaries.
- Do not silently OCR, rasterize, or flatten a searchable PDF. Those operations can lose structure and accessibility.
- For scanned documents, render only the pages needed for the task before expanding to the full file.
- For forms, signatures, annotations, or precise visual layout, extraction alone is insufficient; render relevant pages and inspect them.
- If a dependency is unavailable, state the missing capability and a concrete install option. Do not pretend the document was inspected.

Load `catalog/pdf/REFERENCE.md` with `read_skill_file` only when you need library selection, validation checks, or an advanced operation.
