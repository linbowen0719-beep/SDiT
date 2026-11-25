"""Check the tracked publication surface without printing sensitive matches."""
import ast
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
patterns = {
    'private machine path': re.compile(r'/home/|/data/bowen|/mnt/'),
    'credential-like value': re.compile(r'ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----'),
}
issues = []
for name in filter(None, files):
    p = root / name
    if p.suffix in {'.pt', '.pth', '.safetensors', '.ckpt', '.so', '.pyc'}:
        issues.append((name, 'generated binary/weight'))
    if p.stat().st_size > 10_000_000:
        issues.append((name, 'file exceeds 10 MB'))
    if p.suffix in {'.py', '.json', '.jsonl', '.md', '.txt', '.yml', '.toml'}:
        content = p.read_text()
        if p.suffix == '.py':
            ast.parse(content, filename=name)
        if p.resolve() != Path(__file__).resolve():
            for label, pattern in patterns.items():
                if pattern.search(content):
                    issues.append((name, label))
if issues:
    for name, reason in issues:
        print(f'{name}: {reason}')
    raise SystemExit(1)
print(f'Checked {sum(bool(f) for f in files)} tracked files: no release-surface findings')
