from ultralytics import YOLO


class YOLODetectionTrainer:
    def __init__(
        self,
        yaml_path,
        model_name="yolov12n.pt",
        num_epochs=25,
        image_size=640,
    ):
        """
        Initializes the YOLODetectionTrainer with configuration parameters.
        """
        self.yaml_path = yaml_path
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.model = None
        self.results = None
        self.save_dir = None

    def load_model(self):
        """
        Loads a pre-trained YOLO model.
        """
        print(f"\nLoading pre-trained model: {self.model_name}...")
        self.model = YOLO(self.model_name)
        print("Pre-trained model loaded successfully.")

    def train(self, project="yolo_finetune", name="vehicle_detection", exist_ok=True):
        """
        Fine-tunes the YOLO model on the custom dataset.
        """
        if self.model is None:
            self.load_model()
        print(
            f"\nTraining for {self.num_epochs} epochs with image size {self.image_size}x{self.image_size}."
        )
        print(
            "\nInitiating training... (This may take a while depending on your hardware)"
        )
        self.results = self.model.train(
            data=self.yaml_path,
            epochs=self.num_epochs,
            imgsz=self.image_size,
            project=project,
            name=name,
            exist_ok=exist_ok,
        )
        self.save_dir = self.results.save_dir
        print("Training complete!")
        return self.save_dir
