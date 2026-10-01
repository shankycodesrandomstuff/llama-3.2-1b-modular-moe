"""Fail on obvious credential literals in repository-tracked text files."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "OpenAI key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
}


def main() -> int:
    files = subprocess.check_output(["git", "ls-files", "-z"], text=False).split(b"\0")
    findings: list[str] = []
    for raw_path in filter(None, files):
        path = Path(raw_path.decode())
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path}: possible {name}")
    if findings:
        print("Secret scan failed; remove the detected content before committing.", file=sys.stderr)
        print("\n".join(findings), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
