from pathlib import Path

root = Path("./lucas_data")
lines = []
for cls in sorted(p for p in root.iterdir() if p.is_dir()):
    files = sorted(f.name for f in cls.glob("*.jpg"))
    lines.append(f"[{cls.name}] total files: {len(files)}")
    lines += [f"  {n}" for n in files[:3]]   # first 3 per class = 36 names

Path("filename_sample.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
