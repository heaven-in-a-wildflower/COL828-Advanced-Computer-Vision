import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---- USER INPUTS ----
image_dir = "../../datasets/coco/val2014"
ids_file   = "fixed_ids.txt"
pdf_output = "fixed_images.pdf"

# ---- READ IMAGE IDS ----
with open(ids_file, "r") as f:
    image_ids = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(image_ids)} image IDs.")

# ---- CREATE PDF ----
with PdfPages(pdf_output) as pdf:
    for img_id in image_ids:
        img_id_num = int(img_id)
        
        # Construct filename
        file_name = f"COCO_val2014_000000{img_id_num:06d}.jpg"
        img_path = os.path.join(image_dir, file_name)

        if not os.path.exists(img_path):
            print(f"[WARN] Missing: {img_path}")
            continue

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Plot on a fresh figure
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")

        # Add annotation
        plt.text(
            10, 20, f"ID: {img_id_num}",
            color="yellow",
            fontsize=16,
            bbox=dict(facecolor="black", alpha=0.6)
        )

        # Add page to PDF
        pdf.savefig(bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"Added image ID {img_id_num} to PDF.")

print(f"\nPDF saved as: {pdf_output}")
