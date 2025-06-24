import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import ConfusionMatrixDisplay


class YOLODetectionEvaluator:
    def __init__(self, model_save_dir, yaml_path, class_names):
        """
        Initializes the evaluator with model directory, yaml config, and class names.
        """
        self.model_save_dir = model_save_dir
        self.yaml_path = yaml_path
        self.class_names = class_names
        self.model = None
        self.metrics = None

    def load_best_model(self):
        """
        Loads the best fine-tuned model from the save directory.
        """
        best_model_path = os.path.join(self.model_save_dir, "weights/best.pt")
        self.model = YOLO(best_model_path)
        print(f"Loaded best model from: {best_model_path}")

    def evaluate(self):
        """
        Evaluates the model on the validation set and prints key metrics.
        """
        if self.model is None:
            self.load_best_model()
        self.metrics = self.model.val(data=self.yaml_path)
        print("Validation complete.")
        print("\n--- Key Performance Metrics ---")
        print(f"  mAP50-95: {self.metrics.box.map:.4f}")
        print(f"  mAP50:    {self.metrics.box.map50:.4f}")
        print(f"  mAP75:    {self.metrics.box.map75:.4f}")
        print(f"  Precision: {self.metrics.box.p[0]:.4f}")
        print(f"  Recall:    {self.metrics.box.r[0]:.4f}")
        print("-----------------------------\n")

    def plot_confusion_matrix(self):
        """
        Plots the confusion matrix using scikit-learn.
        """
        if self.metrics is None:
            raise ValueError(
                "You must run evaluate() before plotting the confusion matrix."
            )
        cm_data = self.metrics.confusion_matrix.matrix
        main_cm = cm_data[: len(self.class_names), : len(self.class_names)]
        print("Generating Confusion Matrix plot...")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=main_cm, display_labels=self.class_names
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation="vertical")
        ax.set_title("Confusion Matrix for Vehicle Detection")
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        plt.tight_layout()
        plt.show()
