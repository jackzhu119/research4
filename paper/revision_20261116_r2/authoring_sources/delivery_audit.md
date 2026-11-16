# File delivery audit

The preferred next test is a correctly formatted native output citation for each final DOCX, on its own paragraph. The previous inline citation did not follow the document skill contract. This is a testable host-rendering fix, not proof that downloads work.

## Verified

- `/workspace/Submission.zip`: 20,887,226 bytes, mode 644.
- `/workspace/Manuscript.docx`: 2,661,089 bytes, mode 644.
- `/workspace/Bilingual.docx`: 2,681,021 bytes, mode 644.
- Names are short and English. File permissions do not explain the reported failure.
- `/opt/codex/skills/builtins/documents/SKILL.md`, Final Response Citations, requires each final deliverable citation exactly once, `purpose="output"`, in its own paragraph, never inside prose or a list item. The target must be the verified `sandbox:` URI, with no `mode`.
- Cloud runtime skill says to link the absolute path in the originating chat; **Save as…** downloads a copy on supported desktop clients. It does not guarantee this works on every client.
- Current tool discovery exposes no `open_in_codex`, `attach_artifact`, or `list_artifacts` tool. Thus this agent could not open a native file panel or create an attachment receipt. Earlier reported hangs are not successful attachments.

## Proposed final delivery

After final document validation, return brief prose followed by these standalone directives (actual response must not put them in a code fence):

```text
:codex-file-citation{path="sandbox:/workspace/Manuscript.docx" purpose="output"}

:codex-file-citation{path="sandbox:/workspace/Bilingual.docx" purpose="output"}
```

Use the final verified paths if filenames change. Deliver DOCX files individually before relying on the much larger ZIP. Do not present another raw sandbox Markdown link as an established fix. Do not claim “attached” or “download verified” without a successful attachment result or user confirmation.

If the parent has an actual callable `open_in_codex` tool that this subagent lacks, it may open the existing test manuscript once. A successful panel-open receipt would establish preview access only; it would not establish that the user can save a copy.

## Limitations and fallback

The agent can verify filesystem bytes, formatting syntax, and tool receipts. It cannot inspect the originating chat's native attachment rendering or perform a user-side Save as action with the available tools. No native attachment capability was confirmed in this audit.

A private Google Drive upload could provide an authenticated download route if final-file delivery remains blocked, but it is external delivery, not a native chat attachment. Read the Google Drive skill and obtain parent followup with the final files before uploading; preserve private permissions and do not publish. No upload or external file creation was performed here. No GitHub or server/workspace-path download route is proposed, consistent with the user's preference.
