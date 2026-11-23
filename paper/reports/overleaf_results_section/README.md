# Editable experimental-results section

This is one expanded paper section, not a complete manuscript. It contains
single-model and ensemble results, enlarged qualitative comparisons, a separately
labeled paper-dataset comparison, hyperparameter sensitivity, loss controls,
class-name consolidation and annotation-audit evidence.

Upload the source ZIP through Overleaf: **New Project > Upload Project**. Select
`main.tex` as the main document and **pdfLaTeX** as the compiler. Recompile.

Edit `sections/experimental_results.tex` for the prose, `tables/*.tex` for numeric
tables, and `references.bib` for bibliography entries. Figure files are included.
For a future full manuscript, copy the section, tables, figures and bibliography
entries into the journal's official template. No introduction or full paper is
invented here. Target-journal formatting has not yet been selected.
The explicit column/page breaks near the final audit figures are for this
standalone preview; remove or adjust them when integrating a journal template.

The native benchmark retains confirmed split overlap and annotation errors;
the original-label benchmark also inherits view-name aliases. Published-paper
protocol equivalence is unverified. These limits are retained in the formal text.
Compiled PDFs and intermediate LaTeX files are excluded from the source ZIP.

The source ZIP is sufficient for compilation in Overleaf. To regenerate the
scientific figures from the saved local experiments, run
`python3 tools/prepare_results_section_assets.py` from the repository root.
This also runs `tools/revise_paper_figures.py`; it preserves the manually edited
section prose and bibliography. `revision_provenance.json` records source hashes,
the seven historical sweep settings, qualitative crop selection and limitations.
`tables/hyperparameter_data.csv` contains the plotted sweep observations.

The figure examples are selected to reveal both recovery and failure cases;
they do not estimate how frequently either case occurs. Published-paper protocol
equivalence and accuracy improvements after the r3 corrections remain unverified.
