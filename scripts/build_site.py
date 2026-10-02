"""Inject results.json into the single-file UI -> docs/index.html."""
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
data = (root / "docs" / "results.json").read_text()
html = (root / "site" / "template.html").read_text().replace("__DATA__", data.replace("</", "<\\/"))
(root / "docs" / "index.html").write_text(html)
print("wrote docs/index.html", len(html) // 1024, "KiB")
