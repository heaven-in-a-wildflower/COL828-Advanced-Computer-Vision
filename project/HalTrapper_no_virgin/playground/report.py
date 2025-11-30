import os
import json
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from typing import Dict, Any, List, Optional
import sys

from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log-file-path", type=str, default="results/logs/gdino1.jsonl",
                    help="Path to the generated log file (.jsonl or .yaml)")
args = parser.parse_args()

LOG_FILE_PATH = Path(args.log_file_path)

# derive PDF output path automatically from stem
stem = LOG_FILE_PATH.stem
os.makedirs('results/pdfs',exist_ok=True)
OUTPUT_PDF_PATH = Path("results/pdfs") / f"{stem}.pdf"

print(OUTPUT_PDF_PATH)

# --- Visualization Function (Modified to return the Figure object) ---

def create_inspection_figure(log_data: Dict[str, Any]) -> Optional[plt.Figure]:
    """
    Creates a Matplotlib Figure object containing the image and stats,
    but does NOT call plt.show().
    """
    # extract necessary data from log
    img_id = log_data.get("COCO_id", "N/A")
    response = log_data.get("response", "[NO RESPONSE]").replace(os.linesep, ' ')
    image_path = log_data.get("image")
    
    stats = {
        "CHAIRs": log_data.get("CHAIRs", 0.0),
        "CHAIRi": log_data.get("CHAIRi", 0.0),
        "F1": log_data.get("F1", 0.0),
    }
    hallucinated_words = log_data.get("hallucinated_words", [])

    # console output for progress tracking
    print(f"Processing COCO ID: {img_id} | CHAIR-i: {stats['CHAIRi']:.4f}")
    
    if not image_path or not os.path.exists(image_path):
        print(f"WARNING: Image file not found at {image_path}. Skipping figure generation.", file=sys.stderr)
        return None

    try:
        # create figure with two subplots
        fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1]})

        # --- subplot 1: display the image ---
        img = Image.open(image_path)
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title(f"COCO ID: {img_id}", fontsize=12)

        # --- subplot 2: text/stats with proper wrapping ---
        # wrap the response text to fit within 60 characters per line
        wrapped_response = textwrap.fill(response, width=60)
        
        # wrap hallucinated objects list to fit properly
        hallucinated_str = ', '.join(hallucinated_words) if hallucinated_words else 'None'
        wrapped_objects = textwrap.fill(hallucinated_str, width=60)
        
        text_output = (
            f"VLM Output:\n{'-'*20}\n"
            f"{wrapped_response}\n\n"
            f"CHAIR METRICS:\n{'-'*20}\n"
            f"CHAIR-s: {stats['CHAIRs']}\n"
            f"CHAIR-i: {stats['CHAIRi']:.4f}\n"
            f"F1 Score: {stats['F1']:.4f}\n\n"
            f"HALLUCINATED OBJECTS:\n{wrapped_objects}"
        )
        
        # render text with smaller font to prevent overflow
        ax_text.text(0.05, 0.95, text_output, 
                     transform=ax_text.transAxes, 
                     fontsize=8, 
                     verticalalignment='top', 
                     family='monospace')
        ax_text.axis('off')

        # add main title to the figure
        fig.suptitle(f"VLM Inspection Report - Image ID {img_id}", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    except Exception as e:
        print(f"AN ERROR OCCURRED DURING FIGURE CREATION for ID {img_id}: {e}", file=sys.stderr)
        plt.close(fig)  # ensure figure is closed on error
        return None


# --- Main Function ---

def generate_inspection_report(log_file_path: str, output_pdf_path: str):
    """Loads logs, generates figures, and saves them all to one PDF file."""
    
    if not os.path.exists(log_file_path):
        print(f"FATAL: Log file not found at {log_file_path}.", file=sys.stderr)
        return
        
    try:
        inspection_logs = []
        with open(log_file_path, "r") as f:
            for line in f:
                if line.strip():
                    inspection_logs.append(json.loads(line))
    except Exception as e:
        print(f"ERROR loading logs from {log_file_path}: {e}", file=sys.stderr)
        return

        
    print(f"Loaded {len(inspection_logs)} logs. Generating report to {output_pdf_path}...")
    
    # initialize the PDF writer and save all figures
    with PdfPages(output_pdf_path) as pdf:
        
        for i, log in enumerate(inspection_logs):
            fig = create_inspection_figure(log)
            
            if fig is not None:
                # save the current figure to the PDF file
                pdf.savefig(fig) 
                plt.close(fig)  # close figure to free memory

    print(f"\n✅ Report generation complete. Saved to {output_pdf_path}")


if __name__ == "__main__":
    generate_inspection_report(LOG_FILE_PATH, OUTPUT_PDF_PATH)