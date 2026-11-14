from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageTk


MODEL_PATH = Path(
    r"D:\Medical_Datasets\ChestMNIST\best_chestmnist_model.pth"
)


def load_model(path):
    if not path.is_file():
        raise FileNotFoundError(f"Model not found:\n{path}")

    # Load the trusted checkpoint created by your own training.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    names = list(checkpoint["disease_names"])
    image_size = int(checkpoint["image_size"])

    # Same architecture as your training program.
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        1, 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(names),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()
    return model, image_size, names


def prepare_image(path, image_size):
    with Image.open(path) as original:
        original.load()

        if original.mode not in ("L", "RGB", "RGBA", "P", "1"):
            raise ValueError(
                "Choose an 8-bit PNG or JPG chest X-ray.\n"
                "Raw 16-bit images and DICOM files need "
                "separate preprocessing."
            )

        image = original.convert("L")

    # Your training images were already 128 × 128.
    image = image.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )

    # Exact tensor normalization used during validation.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5],
        ),
    ])

    return image, transform(image).unsqueeze(0)


def predict(model, batch, disease_names):
    with torch.inference_mode():
        scores = torch.sigmoid(model(batch)).flatten()

    if (
        scores.numel() != len(disease_names)
        or not torch.isfinite(scores).all()
    ):
        raise ValueError("The model returned invalid scores.")

    return sorted(
        zip(disease_names, scores.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )


class ModelTester:
    def __init__(self, root, model, image_size, disease_names):
        self.root = root
        self.model = model
        self.image_size = image_size
        self.disease_names = disease_names
        self.preview_image = None

        root.title("Test your ChestMNIST model")
        root.geometry("850x590")
        root.minsize(720, 540)

        page = ttk.Frame(root, padding=16)
        page.pack(fill="both", expand=True)

        ttk.Label(
            page,
            text="Chest X-ray model tester",
            font=("Segoe UI", 17, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            page,
            text="Your saved model is loaded. Choose an X-ray to test.",
        ).pack(anchor="w", pady=(6, 12))

        self.choose_button = ttk.Button(
            page,
            text="Choose X-ray",
            command=self.choose_image,
        )
        self.choose_button.pack(anchor="w")

        self.filename = ttk.Label(
            page,
            text="No image selected",
            wraplength=760,
        )
        self.filename.pack(anchor="w", pady=(8, 12))

        body = ttk.Frame(page)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, width=300)
        left.pack(side="left", fill="y", padx=(0, 20))
        left.pack_propagate(False)

        ttk.Label(
            left,
            text="Image supplied to the model",
        ).pack(anchor="w", pady=(0, 8))

        self.preview = ttk.Label(
            left,
            text="Your selected image appears here",
            anchor="center",
        )
        self.preview.pack(fill="both", expand=True)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        self.table = ttk.Treeview(
            right,
            columns=("finding", "score"),
            show="headings",
            height=14,
        )

        self.table.heading("finding", text="Finding label")
        self.table.heading("score", text="Model score (0–1)")

        self.table.column(
            "finding", width=220, minwidth=170
        )
        self.table.column(
            "score", width=130, minwidth=110, anchor="center"
        )

        self.table.pack(fill="both", expand=True)

        ttk.Label(
            page,
            text=(
                "Research scores only; these are not diagnoses "
                "or calibrated clinical probabilities. "
                "Labels are independent, so scores do not "
                "need to sum to 1."
            ),
            wraplength=760,
        ).pack(anchor="w", pady=(12, 6))

        self.status = tk.StringVar(
            value="Ready. No training is required."
        )

        ttk.Label(
            page,
            textvariable=self.status,
            wraplength=760,
        ).pack(anchor="w")

    def choose_image(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Choose an 8-bit chest X-ray",
            filetypes=[
                (
                    "X-ray images",
                    "*.png *.jpg *.jpeg *.bmp",
                ),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        self.table.delete(*self.table.get_children())
        self.filename.config(text=Path(path).name)
        self.preview_image = None
        self.preview.config(image="", text="Reading image...")
        self.choose_button.config(state="disabled")
        self.status.set("Running your saved model...")
        self.root.update_idletasks()

        try:
            image, batch = prepare_image(
                path,
                self.image_size,
            )

            results = predict(
                self.model,
                batch,
                self.disease_names,
            )

            self.preview_image = ImageTk.PhotoImage(
                image.resize(
                    (280, 280),
                    Image.Resampling.NEAREST,
                )
            )

            self.preview.config(
                image=self.preview_image,
                text="",
            )

            for disease, score in results:
                self.table.insert(
                    "",
                    "end",
                    values=(disease, f"{score:.4f}"),
                )
                print(f"{disease:25s} {score:.4f}")

            self.status.set(
                "Prediction complete. Choose another X-ray "
                "to test again."
            )

        except Exception as error:
            self.preview.config(
                image="",
                text="Image could not be tested",
            )
            self.status.set(
                "Choose another image or check the error message."
            )
            messagebox.showerror(
                "Could not test image",
                str(error),
                parent=self.root,
            )

        finally:
            self.choose_button.config(state="normal")


def main():
    torch.set_num_threads(2)

    root = tk.Tk()
    root.withdraw()

    try:
        print("Loading your saved model:", MODEL_PATH)

        model, image_size, disease_names = load_model(
            MODEL_PATH
        )

    except Exception as error:
        messagebox.showerror(
            "Could not load model",
            str(error),
            parent=root,
        )
        root.destroy()
        return

    ModelTester(
        root,
        model,
        image_size,
        disease_names,
    )

    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()