import fiftyone as fo
import fiftyone.zoo as foz


class CocoDatasetDownloader:
    def __init__(
        self,
        classes=None,
        train_samples=2500,
        val_samples=500,
        dataset_dir="../data/coco_vehicles",
    ):
        self.classes = classes if classes is not None else ["car"]
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.dataset_dir = dataset_dir
        self.train_dataset = None
        self.val_dataset = None

    def download(self):
        print("\nDownloading and preparing the dataset...")
        print(f"Target classes: {self.classes}")
        print(f"Dataset will be stored in: {self.dataset_dir}")
        print("Downloading training split...")
        self.train_dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="train",
            label_types=["detections"],
            classes=self.classes,
            max_samples=self.train_samples,
        )
        print(f"Training dataset summary:\n{self.train_dataset}")
        print("\nDownloading validation split...")
        self.val_dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            label_types=["detections"],
            classes=self.classes,
            max_samples=self.val_samples,
        )
        print(f"Validation dataset summary:\n{self.val_dataset}")

    def export(self):
        print("\nExporting datasets to YOLO format...")
        self.train_dataset.export(
            export_dir=self.dataset_dir,
            dataset_type=fo.types.YOLOv5DatasetType,
            label_field="ground_truth",
            split="train",
            classes=self.classes,
        )
        self.val_dataset.export(
            export_dir=self.dataset_dir,
            dataset_type=fo.types.YOLOv5DatasetType,
            label_field="ground_truth",
            split="val",
            classes=self.classes,
        )
        print("Dataset prepared and exported successfully.")
        return self.dataset_dir, self.classes

    def download_and_prepare(self):
        self.download()
        return self.export()
