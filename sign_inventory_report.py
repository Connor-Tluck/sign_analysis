import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import numpy as np


def count_nonempty_text(series):
    """Helper to count how many rows have non-empty text."""
    return series.fillna("").str.strip().ne("").sum()


def autocrop_alpha(img):
    """
    If 'img' has an alpha channel, crop out fully-transparent areas.
    Returns the cropped image. If no alpha channel or fully opaque, returns original.

    img is assumed to be a numpy array of shape (H, W, 4) for RGBA.
    """
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        # Create a boolean mask where alpha > 0
        mask = alpha > 0
        if not np.any(mask):
            # If the entire image is transparent, return original
            return img
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Crop
        return img[rmin:rmax + 1, cmin:cmax + 1, :]
    else:
        # No alpha channel or not RGBA
        return img


def generate_sign_image_table_report(
        input_csv,
        output_pdf="sign_image_table_report.pdf",
        logo_path=None,
        image_folder=None,
        rows_per_page=10
):
    """
    Reads a CSV of sign data (with columns like MUTCD, Text, Width, Height, Orientation, etc.),
    summarizes by MUTCD, then creates a multi-page PDF with a clean table layout:
      - Each row includes a small, centered PNG image (if found) + text columns.
      - Black lines around columns and rows, with a small margin so nothing overlaps the grid.
      - If the PNG has an alpha channel, automatically crops out fully transparent areas
        so the visible sign is truly centered.

    Parameters:
      input_csv (str): Path to the input CSV file.
      output_pdf (str): Path for the output PDF file.
      logo_path (str): Optional path to a Mach9 logo image for the cover page.
      image_folder (str): Folder containing PNG images named by MUTCD code (e.g., "D3-1.png").
      rows_per_page (int): Number of table rows per page before starting a new page.
    """

    # 1. Read CSV
    df = pd.read_csv(input_csv, low_memory=False)

    # 2. Gather cover info
    dataset_name = os.path.basename(input_csv)
    total_rows = len(df)
    date_processed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 3. Aggregate by MUTCD
    group_obj = df.groupby("MUTCD", dropna=False)
    summary_df = group_obj.agg(
        Count=("MUTCD", "size"),
        TextCount=("Text", count_nonempty_text),
        AvgWidth=("Width", "mean"),
        AvgHeight=("Height", "mean"),
        AvgOrientation=("Orientation", "mean")
    ).reset_index()

    # Sort descending by Count
    summary_df.sort_values(by="Count", ascending=False, inplace=True)

    # Round numeric columns
    numeric_cols = ["AvgWidth", "AvgHeight", "AvgOrientation"]
    summary_df[numeric_cols] = summary_df[numeric_cols].round(2)

    unique_mutcd_count = summary_df["MUTCD"].nunique()

    # Define columns (including the "image" column)
    columns = [
        {"key": "image", "header": "Image", "width": 0.15},
        {"key": "MUTCD", "header": "MUTCD", "width": 0.15},
        {"key": "Count", "header": "Count", "width": 0.10},
        {"key": "TextCount", "header": "# With Text", "width": 0.10},
        {"key": "AvgWidth", "header": "Avg Width", "width": 0.10},
        {"key": "AvgHeight", "header": "Avg Height", "width": 0.10},
        {"key": "AvgOrientation", "header": "Avg Orientation", "width": 0.15},
    ]

    # How many pages do we need?
    n = len(summary_df)
    pages = (n + rows_per_page - 1) // rows_per_page

    header_fontsize = 9
    cell_fontsize = 8
    cell_margin = 0.05  # fraction of each cell's width/height used as margin

    with PdfPages(output_pdf) as pdf:
        # ------------------ COVER PAGE ------------------
        fig_cover = plt.figure(figsize=(8.27, 11.69))
        ax_cover = fig_cover.add_axes([0, 0, 1, 1])
        ax_cover.axis('off')

        title_text = "Mach9 - MUTCD Sign Table Report"
        cover_text = f"""
{title_text}

Input CSV: {dataset_name}
Date Processed: {date_processed}

Total Rows: {total_rows}
Unique MUTCD Codes: {unique_mutcd_count}
        """
        ax_cover.text(
            0.5, 0.5, cover_text,
            ha='center', va='center', fontsize=14
        )

        # Optional Mach9 logo
        if logo_path and os.path.exists(logo_path):
            ax_logo = fig_cover.add_axes([0.35, 0.7, 0.3, 0.2])
            logo_img = plt.imread(logo_path)
            ax_logo.imshow(logo_img)
            ax_logo.axis('off')

        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        # ------------------ TABLE PAGES ------------------
        for page_idx in range(pages):
            start_i = page_idx * rows_per_page
            end_i = min(start_i + rows_per_page, n)
            chunk_df = summary_df.iloc[start_i:end_i]

            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')

            # Page title
            ax.text(0.5, 0.95, f"Sign Summary (Page {page_idx + 1}/{pages})",
                    ha='center', va='bottom', fontsize=12)

            # Table region
            left_margin = 0.05
            right_margin = 0.95
            top_margin = 0.88
            bottom_margin = 0.07

            table_width = right_margin - left_margin
            table_height = top_margin - bottom_margin

            # We'll have 1 header row + rows_per_page data rows
            row_height = table_height / (rows_per_page + 1)
            # Sum of column widths is 1.0 (or less/more). We'll distribute actual space accordingly
            total_col_fraction = sum(col["width"] for col in columns)

            # x_positions: boundaries of each column
            x_positions = [left_margin]
            for col in columns:
                x_positions.append(x_positions[-1] + col["width"] * table_width / total_col_fraction)

            # ------------- DRAW HEADER -------------
            header_y_top = top_margin
            header_y_bottom = header_y_top - row_height

            # Horizontal lines for header
            ax.plot([left_margin, right_margin], [header_y_top, header_y_top],
                    color="black", linewidth=1)
            ax.plot([left_margin, right_margin], [header_y_bottom, header_y_bottom],
                    color="black", linewidth=1)
            # Vertical lines
            for x in x_positions:
                ax.plot([x, x], [header_y_top, header_y_bottom], color="black", linewidth=1)

            # Place header text
            for col_idx, col in enumerate(columns):
                col_x0 = x_positions[col_idx]
                col_x1 = x_positions[col_idx + 1]
                cx = (col_x0 + col_x1) / 2
                cy = (header_y_top + header_y_bottom) / 2
                ax.text(cx, cy, col["header"], ha="center", va="center", fontsize=header_fontsize)

            # ------------- DRAW ROWS -------------
            for row_i, row_data in enumerate(chunk_df.itertuples()):
                row_y_top = header_y_bottom - row_i * row_height
                row_y_bottom = row_y_top - row_height

                # Horizontal lines
                ax.plot([left_margin, right_margin], [row_y_top, row_y_top],
                        color="black", linewidth=1)
                ax.plot([left_margin, right_margin], [row_y_bottom, row_y_bottom],
                        color="black", linewidth=1)
                # Vertical lines
                for x in x_positions:
                    ax.plot([x, x], [row_y_top, row_y_bottom], color="black", linewidth=1)

                # Fill cells in this row
                for col_idx, col in enumerate(columns):
                    col_x0 = x_positions[col_idx]
                    col_x1 = x_positions[col_idx + 1]
                    cell_width = (col_x1 - col_x0)
                    cell_height = (row_y_top - row_y_bottom)

                    # Sub-axes bounding box with margin
                    ax_left = col_x0 + cell_margin * cell_width
                    ax_bottom = row_y_bottom + cell_margin * cell_height
                    ax_width = cell_width * (1 - 2 * cell_margin)
                    ax_height = cell_height * (1 - 2 * cell_margin)

                    # Create a sub-axes for cell content
                    cell_ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
                    cell_ax.axis('off')
                    cell_ax.set_aspect("equal", "box")

                    if col["key"] == "image":
                        mutcd_code = str(row_data.MUTCD)
                        png_path = None
                        if image_folder:
                            candidate_path = os.path.join(image_folder, mutcd_code + ".png")
                            if os.path.exists(candidate_path):
                                png_path = candidate_path
                            else:
                                print(f"[WARN] PNG not found for {mutcd_code}: {candidate_path}")

                        if png_path:
                            try:
                                img = plt.imread(png_path)
                                img = autocrop_alpha(img)  # auto-crop transparent edges if RGBA
                                cell_ax.imshow(img)
                            except Exception as e:
                                print(f"[WARN] Failed to load/crop image for {mutcd_code}: {e}")
                    else:
                        # For text columns
                        if col["key"] == "MUTCD":
                            cell_text = str(row_data.MUTCD)
                        elif col["key"] == "Count":
                            cell_text = str(row_data.Count)
                        elif col["key"] == "TextCount":
                            cell_text = str(row_data.TextCount)
                        elif col["key"] == "AvgWidth":
                            cell_text = str(row_data.AvgWidth)
                        elif col["key"] == "AvgHeight":
                            cell_text = str(row_data.AvgHeight)
                        elif col["key"] == "AvgOrientation":
                            cell_text = str(row_data.AvgOrientation)
                        else:
                            cell_text = ""

                        # Place text at center of the cell sub-axes
                        cell_ax.text(0.5, 0.5, cell_text,
                                     ha="center", va="center", fontsize=cell_fontsize)

            pdf.savefig(fig)
            plt.close(fig)

    print(f"PDF report saved to: {output_pdf}")


# ---------------- Example usage ----------------
if __name__ == "__main__":
    # CSV with columns: MUTCD, Text, Width, Height, Orientation, etc.
    input_csv_file = r"sign-reporting/Sign Face.csv"
    # Output PDF
    output_pdf_file = r"sign-reporting/sign_image_table_report.pdf"
    # Mach9 logo (optional)
    logo_path = r"Logos\Mach9_Logo_Black 1.png"
    # Folder with PNG images named e.g. "D3-1.png", "W1-1.png", etc.
    image_folder = r"mutcd_signs"

    # Generate the report
    generate_sign_image_table_report(
        input_csv=input_csv_file,
        output_pdf=output_pdf_file,
        logo_path=logo_path,
        image_folder=image_folder,
        rows_per_page=10  # Adjust as needed
    )
