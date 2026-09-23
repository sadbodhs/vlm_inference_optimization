#!/usr/bin/env python3
"""Stage the repo's markdown for MkDocs without duplicating anything in git.

Committed docs use relative links that are correct when browsing the repo on
GitHub (../scripts/foo.sh, ../README.md). On a published site those resolve
outside the site root and 404. This rewrites them at BUILD time only:

  docs/index.md        -> index.md                  (the site LANDS on the question)
  ../README.md         -> overview.md               (README is a page, not the root)
  ../<non-md path>     -> absolute GitHub blob URL
  docs/foo.md (README) -> foo.md

GitHub always shows README.md first and there is no changing that, but a site
visitor has no such constraint: someone arriving cold is better served by "here is
the question" than by a table of numbers.

Committed files are never modified.
"""
import os
import re
import shutil
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "site_src"
BLOB = "https://github.com/sadbodhs/vlm_inference_optimization/blob/main"
README_PAGE = "overview.md"

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

link_re = re.compile(r"\]\(([^)]+)\)")


def _external(href):
    return href.startswith(("http://", "https://", "#", "mailto:"))


def fix_docs_link(m):
    href = m.group(1)
    if _external(href):
        return m.group(0)
    target, hashmark, frag = href.partition("#")
    if target == "../README.md":
        return f"]({README_PAGE}{hashmark}{frag})"
    if href.startswith("../"):
        # anything outside docs/ is not in the site tree -> send it to GitHub
        return f"]({BLOB}/{href[3:]})"
    return m.group(0)


def fix_readme_link(m):
    href = m.group(1)
    if _external(href):
        return m.group(0)
    target, hashmark, frag = href.partition("#")
    if target == "docs/index.md":
        return f"](index.md{hashmark}{frag})"
    if target.startswith("docs/") and target.endswith(".md"):
        return f"]({target[5:]}{hashmark}{frag})"
    if target.startswith("docs/"):
        # an asset under docs/ (a figure) is copied to the site root with the pages;
        # a GitHub blob URL would be an HTML page, not an image
        return f"]({target[5:]}{hashmark}{frag})"
    return f"]({BLOB}/{href})"


# docs/*.md -> site root
docs = os.path.join(REPO, "docs")
for root, _, files in os.walk(docs):
    rel = os.path.relpath(root, docs)
    dest = OUT if rel == "." else os.path.join(OUT, rel)
    os.makedirs(dest, exist_ok=True)
    for f in files:
        src = os.path.join(root, f)
        if f.endswith(".md"):
            text = open(src, encoding="utf-8").read()
            open(os.path.join(dest, f), "w", encoding="utf-8").write(
                link_re.sub(fix_docs_link, text))
        else:
            shutil.copy2(src, os.path.join(dest, f))

# README.md -> overview.md
readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
open(os.path.join(OUT, README_PAGE), "w", encoding="utf-8").write(
    link_re.sub(fix_readme_link, readme))

print(f"staged {sum(len(fs) for _, _, fs in os.walk(OUT))} files into {OUT}/")
