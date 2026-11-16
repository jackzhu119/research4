# Final localized update package audit

**PASS** — independent read-only audit of `Update.zip` on 16 November 2026. No source/package edits, uploads or agent spawning.

Package: 20,900,213 bytes; 49 unique entries; SHA-256 `f8efc63fae3a7e2ddbd0d795b2e082fe3d68f04af6a46f84f8f81d122292811a`. Values match the current final package receipt. Archive CRC passes. All 48 manifest-listed members match declared sizes/hashes and their current repository source bytes; coverage includes every archive member except the manifest itself.

All requested submission/read files are present. Every DOCX/XLSX internal archive passes CRC. Six XLSX supplements remain below 10 MiB. Six main EPS figures and supplementary EPS retain valid signatures, bounding boxes and fonts and match source bytes.

Against baseline commit `64d95fe4371521dd3c16cc0ee792a82332b570be`, independent canonical-XML comparison confirms exactly one inserted English manuscript paragraph, two inserted bilingual counterpart paragraphs, and one replaced supplementary paragraph. Only `word/document.xml` changed in these three DOCX archives; every other OOXML part is byte-identical. Source-block comparison confirms only `r3_errors` and `s5_errors` changed. The abstract source is exactly unchanged, and all unrelated document XML is unchanged.

The other four DOCX files (title page, cover letter, declarations and revision notes), all packaged EPS figures and all XLSX supplements are byte-identical to baseline. Original author/header/end-matter XML matches the metadata lock in both manuscripts; AI disclosure occurs exactly once in each. Auxiliary author, approval, AI and signature fields retain the authorized baseline values.

Current artifact/visual receipts are PASS and their seven DOCX hashes match final files. All 77 page indices are covered, and all 77 recorded PNG hashes match current review3 renders or unchanged review2/render2 renders. The localized patch, Shanxi evidence check and changed-main visual-review receipts are PASS. Direct review of changed pages is attributed to the current visual receipts; this package audit independently verifies their file/hash correspondence.

No defects found. This verifies the localized package and preservation boundary; it does not establish user-side download success or journal submission.
