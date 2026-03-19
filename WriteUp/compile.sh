#!/usr/bin/env bash

set -e  # stop on error

MAIN="main"
BUILD_DIR="build"

mkdir -p "$BUILD_DIR"

echo "compiling"

# first pass
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" "$MAIN.tex"

# bib
bibtex "$BUILD_DIR/$MAIN"

# cecond pass
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" "$MAIN.tex"

# third pass (to fix refs)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" "$MAIN.tex"

echo "done"
echo "PDF: $BUILD_DIR/$MAIN.pdf"
