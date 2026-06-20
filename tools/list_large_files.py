import os

files = []
for root, dirs, names in os.walk('.', topdown=True):
    for n in names:
        p = os.path.join(root, n)
        try:
            s = os.path.getsize(p)
        except OSError:
            continue
        files.append((s, p))

files.sort(reverse=True)
for s, p in files[:40]:
    print(f"{s}\t{p}")
