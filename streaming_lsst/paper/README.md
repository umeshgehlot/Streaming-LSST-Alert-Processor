# Streaming LSST Alert Processor - Research Paper

This directory contains the LaTeX source code and generated figures for the research paper: **"Real-Time Anomaly Detection in Time-Domain Astronomy: A Streaming Graph Transformer Approach for the LSST Era"**.

## Files
- `main.tex`: The main LaTeX document (IEEEtran format).
- `references.bib`: Bibliography file containing all citations.
- `generate_plots.py`: Python script used to generate the evaluation figures from benchmark results.
- `figures/`: Directory containing the generated `.png` and `.pdf` plots.

## How to Compile

Since a local LaTeX distribution (like `pdflatex` or `latexmk`) might not be installed on your system, the easiest way to compile and view the paper is using Overleaf:

### Option 1: Overleaf (Recommended)
1. Go to [Overleaf](https://www.overleaf.com/) and create a new blank project.
2. Upload `main.tex` and `references.bib` to the root of the project.
3. Create a folder named `figures` in Overleaf.
4. Upload the images from the local `paper/figures/` directory into the Overleaf `figures` folder.
5. Click **Recompile**. Overleaf will automatically handle the IEEEtran formatting and generate the PDF.

### Option 2: Local TeX Distribution (MiKTeX / TeX Live)
If you install a LaTeX distribution locally, open a terminal in this `paper` directory and run:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

This will generate `main.pdf`.

## Editing the Paper
- The document uses the `IEEEtran` class.
- You can add author names and affiliations at the top of `main.tex` under the `\author{}` block.
- To regenerate the plots with different parameters, simply run `python generate_plots.py` from the project root.
