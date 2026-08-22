from pathlib import Path
import os
from urllib.request import urlopen

import cv2
import numpy as np


class SceneClassification:
    WEIGHTS_URL = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
    CATEGORIES_URL = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
    CACHE_DIRECTORY = Path(os.getenv("POROTTA_PLACES365_CACHE", "/tmp/porotta/places365"))
    OUTDOOR_TERMS = {
        "airport_terminal", "beach", "bridge", "campus", "coast", "construction_site", "desert",
        "field", "forest", "garden", "gas_station", "harbor", "highway", "lake", "mountain",
        "outdoor", "parking_lot", "park", "pier", "playground", "plaza", "railroad", "river",
        "road", "sky", "stadium", "street", "swimming_pool", "train_station", "valley", "village",
        "waterfall",
    }

    def __init__(self) -> None:
        self.model = None
        self.transform = None
        self.labels = None

    def execute(self, frame: np.ndarray) -> dict:
        self._load()
        import torch
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
            values, indices = torch.topk(probabilities, 5)
        predictions = [
            {"label": self.labels[int(index)], "confidence": round(float(value), 4)}
            for value, index in zip(values, indices)
        ]
        top_class = predictions[0]
        return {
            "scene_classification": {
                "category": self._category(top_class["label"]),
                "top_class": top_class,
                "predictions": predictions,
            }
        }

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms
        except ImportError as error:
            raise RuntimeError("Layer 10 requires torch and torchvision") from error

        self.CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        category_path = self.CACHE_DIRECTORY / "categories_places365.txt"
        if not category_path.exists():
            with urlopen(self.CATEGORIES_URL, timeout=60) as response:
                category_path.write_bytes(response.read())
        self.labels = self._read_labels(category_path)
        self.model = models.resnet18(weights=None, num_classes=365)
        torch.hub.set_dir(str(self.CACHE_DIRECTORY / "torch"))
        checkpoint = torch.hub.load_state_dict_from_url(
            self.WEIGHTS_URL,
            map_location="cpu",
            progress=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _read_labels(self, path: Path) -> list[str]:
        labels = []
        for line in path.read_text().splitlines():
            label = line.strip().split()[0].lstrip("/")
            labels.append(label[2:] if len(label) > 2 and label[1] == "/" else label)
        if len(labels) != 365:
            raise RuntimeError(f"Places365 category file contains {len(labels)} labels, expected 365")
        return labels

    def _category(self, label: str) -> str:
        normalized = label.lower().replace(" ", "_")
        return "outdoor" if any(term in normalized for term in self.OUTDOOR_TERMS) else "indoor"
