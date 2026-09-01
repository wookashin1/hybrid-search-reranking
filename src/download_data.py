"""Skini i raspakuj BEIR dataset u data/<naziv>/.
Koristi:  python src/download_data.py [scifact]
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def download(name="scifact", out_dir="data"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    if (out / name).exists():
        print(f"{out/name} vec postoji — preskacem")
        return

    url = f"{BASE}/{name}.zip"
    print(f"skidam {url} ...")
    data = urllib.request.urlopen(url).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(out)
    print(f"raspakovano u {out/name}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    download(name)
