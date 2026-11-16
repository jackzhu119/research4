# Second review builder changes

Edited only build_documents.py and added REVIEW2_DECISIONS.json plus this report. No main/SI content, metadata lock, template, raw data or final DOCX was changed or generated.

- Protected URLs, DOI strings, email, ORCID, hashes, source IDs, hyphenated instrument IDs and numerical thousands before prose spacing. Fixed glued English counts, semicolon/list/set commas, and pp after intervals. Editable table text also uses the same typography function. Locked author/declaration XML remains on its original bypass path.
- Compact tables (at most eight data rows and under 1,600 text characters) keep their rows together. Longer tables retain repeating headers and keep first/last data-row pairs together; individual rows cannot split. Layout still requires final render inspection.
- Replaced first-round REVISION_ROWS with twelve second-round accept/modify/reject decisions, in English and Chinese. REVIEW2_DECISIONS.json preserves the same rationale and names the two review inputs and audit sources.
- Replaced the notes introduction and evidence section with concise second-round narrative. Removed stale task-count/project-log details. Decisions distinguish SVM gamma-cache scope from non-SVM background baseline eligibility, retain the matched-SVM gain and strong LDA, qualify fixed-PRED/R0, explain descriptive source confusion, avoid unproved kernel mechanisms/procurement, and distinguish the requested abstract ceiling from archived journal guidance.

Validation: parsed the entire builder using ast.parse; executed only its extracted tidy function against exact expectations for thousands, set/list punctuation, interval pp, DOI URL, ORCID and instrument/source IDs. All checks passed. Did not import the builder, run Pandoc, or generate DOCX. Root should run its authoring marker before final generation and inspect pagination after rendering.
