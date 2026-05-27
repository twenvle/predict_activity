from pathlib import Path
import glob

base = Path(__file__).resolve().parent.parent.parent
print(base)

path = base / f"data/in/unknown/"
print(path)
path = list(path.glob("*/logfile/sub_88-99-3.log"))[0]
print(path)
