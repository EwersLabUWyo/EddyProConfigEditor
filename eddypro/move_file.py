# put each "chunk" of data in its own directory
from pathlib import Path
def _copy(self, target):
    import shutil
    assert self.is_file()
    shutil.copy(str(self), str(target))  # str() only there for Python < (3, 6)

Path.copy = _copy

from datetime import datetime

raw_data_dir = Path('/Users/alex/Documents/Data/Platinum_EC/LostCreek/biomet')
raw_files = list(raw_data_dir.glob('biomet_*.data'))
for f in raw_files:
    file_timestamp = datetime.strptime(f.name[-13:-5], r'%Y%m%d')
    ymd = f'{file_timestamp.year:04d}_{file_timestamp.month:02d}_{file_timestamp.day:02d}'
    print(ymd)
    Path.mkdir(raw_data_dir / ymd, exist_ok=True)
    f.copy(f.parent / ymd / f.name)
