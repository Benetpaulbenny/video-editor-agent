import shutil
from pathlib import Path


class StorageManager:
    def __init__(self, root: str = "~/Downloads/Pookie") -> None:
        self.root = Path(root).expanduser()

    def reset(self) -> None:
        if self.root.name != "Pookie":
            raise ValueError("Storage root must be the Pookie folder")
        if not self.root.exists():
            return
        for item in self.root.iterdir():
            if item.is_symlink() or item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
