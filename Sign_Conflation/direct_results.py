import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from datetime import datetime


def generate_direct_results(df_mach9, df_client, logo_path, output_pdf, duplicates_info=None):
    # Ensure both DataFrames have unified column names:
    # df_mach9 is assumed to have "Latitude", "Longitude", and "MUTCD"
    # If df_client does not have a "MUTCD" column, create a dummy one.
    if "MUTCD" not in df_client.columns:
        df_client = df_client.copy()
        df_client["MUTCD"] = "N/A"

    with PdfPages(output_pdf) as pdf:
        # COVER PAGE
        fig_cover = plt.figure(figsize=(8.27, 11.69))
        ax_cover = fig_cover.add_axes([0, 0, 1, 1])
        ax_cover.axis("off")
        title = "Mach9 Direct Results"
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cover_text = f"{title}\n\nReport Generated: {gen_time}"
        if duplicates_info:
            cover_text += (
                f"\n\nDuplicate Removal (Mach9): Orig={duplicates_info['File 1']['original_count']}, "
                f"Removed={duplicates_info['File 1']['duplicates_removed']}, Unique={duplicates_info['File 1']['unique_count']}\n"
                f"Client: Orig={duplicates_info['File 2']['original_count']}, "
                f"Removed={duplicates_info['File 2']['duplicates_removed']}, Unique={duplicates_info['File 2']['unique_count']}"
            )
        ax_cover.text(0.5, 0.5, cover_text, fontsize=16, ha="center", va="center")
        if logo_path and os.path.exists(logo_path):
            ax_logo = fig_cover.add_axes([0.35, 0.7, 0.3, 0.2])
            logo = mpimg.imread(logo_path)
            ax_logo.imshow(logo)
            ax_logo.axis("off")
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        # PAGE 1: Comparison Charts (2×2 grid)
        fig1 = plt.figure(figsize=(8.27, 11.69))
        gs1 = GridSpec(nrows=2, ncols=2, figure=fig1, wspace=0.3, hspace=0.3)
        # Chart 1: Total Sign Count Comparison
        ax1 = fig1.add_subplot(gs1[0, 0])
        totals = [len(df_mach9), len(df_client)]
        ax1.bar(["Mach9", "Client"], totals, color=["blue", "orange"])
        ax1.set_title("Total Sign Count")
        for i, v in enumerate(totals):
            ax1.text(i, v + 0.5, str(v), ha="center", va="bottom")
        # Chart 2: Unique MUTCD Codes (Mach9 only)
        ax2 = fig1.add_subplot(gs1[0, 1])
        unique_mach9 = len(set(df_mach9["MUTCD"].unique()) - set(df_client["MUTCD"].unique()))
        ax2.bar(["Mach9 Unique"], [unique_mach9], color=["green"])
        ax2.set_title("Unique MUTCD Codes (Mach9)")
        ax2.text(0, unique_mach9 + 0.5, str(unique_mach9), ha="center", va="bottom")
        # Chart 3: Overlay Map
        ax3 = fig1.add_subplot(gs1[1, 0])
        if "Longitude" in df_mach9.columns and "Latitude" in df_mach9.columns:
            ax3.scatter(df_mach9["Longitude"], df_mach9["Latitude"], color="blue", s=5, label="Mach9")
        if "Longitude" in df_client.columns and "Latitude" in df_client.columns:
            ax3.scatter(df_client["Longitude"], df_client["Latitude"], color="orange", s=5, label="Client")
        ax3.set_title("Map Overlay")
        ax3.set_xlabel("Longitude")
        ax3.set_ylabel("Latitude")
        ax3.legend()
        ax3.grid(True)
        # Chart 4: False Positives (client codes not in Mach9)
        ax4 = fig1.add_subplot(gs1[1, 1])
        false_pos = set(df_client["MUTCD"].unique()) - set(df_mach9["MUTCD"].unique())
        ax4.bar(["False Positives"], [len(false_pos)], color="red")
        ax4.set_title("False Positives (Client)")
        pdf.savefig(fig1)
        plt.close(fig1)

        # PAGE 2: Map Comparison
        fig2, ax2 = plt.subplots(figsize=(8.27, 11.69))
        ax2.scatter(df_mach9["Longitude"], df_mach9["Latitude"], color="blue", s=5, label="Mach9")
        ax2.scatter(df_client["Longitude"], df_client["Latitude"], color="orange", s=5, label="Client")
        ax2.set_title("Map Comparison: Mach9 vs. Client")
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        ax2.legend()
        ax2.grid(True)
        pdf.savefig(fig2)
        plt.close(fig2)

        # PAGE 3: Temporal Analysis (if Timestamp exists)
        if "Timestamp" in df_mach9.columns and "Timestamp" in df_client.columns:
            df_mach9["Timestamp"] = pd.to_datetime(df_mach9["Timestamp"])
            df_client["Timestamp"] = pd.to_datetime(df_client["Timestamp"])
            mach9_time = df_mach9.resample("M", on="Timestamp").size()
            client_time = df_client.resample("M", on="Timestamp").size()
            fig3, ax3 = plt.subplots(figsize=(8.27, 11.69))
            ax3.plot(mach9_time.index, mach9_time.values, label="Mach9", marker="o")
            ax3.plot(client_time.index, client_time.values, label="Client", marker="o")
            ax3.set_title("Monthly Sign Count Over Time")
            ax3.set_xlabel("Month")
            ax3.set_ylabel("Count")
            ax3.legend()
            pdf.savefig(fig3)
            plt.close(fig3)

        # PAGE 4: Confidence Scores Comparison (if Confidence exists)
        if "Confidence" in df_mach9.columns or "Confidence" in df_client.columns:
            data = []
            labels = []
            if "Confidence" in df_mach9.columns:
                data.append(df_mach9["Confidence"].dropna())
                labels.append("Mach9")
            if "Confidence" in df_client.columns:
                data.append(df_client["Confidence"].dropna())
                labels.append("Client")
            fig4, ax4 = plt.subplots(figsize=(8.27, 11.69))
            ax4.boxplot(data, labels=labels)
            ax4.set_title("Confidence Scores Comparison")
            pdf.savefig(fig4)
            plt.close(fig4)

        # PAGE 5: Final Summary and Aggregated MUTCD Table (Mach9 only)
        df_group = df_mach9.groupby("MUTCD", dropna=False)
        total_mach9 = len(df_mach9)
        summary_df = pd.DataFrame({
            "MUTCD": df_group["MUTCD"].first(),
            "Count": df_group.size(),
            "# With Text": df_group["HasText"].sum() if "HasText" in df_mach9.columns else 0,
            "Avg Width": df_group["Width"].mean() if "Width" in df_mach9.columns else 0,
            "Avg Height": df_group["Height"].mean() if "Height" in df_mach9.columns else 0,
            "Avg Orientation": df_group["Orientation"].mean() if "Orientation" in df_mach9.columns else 0
        }).reset_index(drop=True)
        summary_df["% of total"] = (summary_df["Count"] / total_mach9) * 100
        summary_df = summary_df.sort_values("Count", ascending=False).reset_index(drop=True)
        fig5 = plt.figure(figsize=(8.27, 11.69))
        ax5 = fig5.add_axes([0.1, 0.1, 0.8, 0.8])
        ax5.axis("off")
        summary_text = f"Final Summary\n\nTotal Mach9 Signs: {total_mach9}\n"
        ax5.text(0.5, 0.9, summary_text, ha="center", va="center", fontsize=14)
        pdf.savefig(fig5)
        plt.close(fig5)
        rows_per_page = 30
        total_rows = len(summary_df)
        num_pages = math.ceil(total_rows / rows_per_page)
        for page_idx in range(num_pages):
            figT = plt.figure(figsize=(8.27, 11.69))
            gsT = GridSpec(nrows=rows_per_page, ncols=8, figure=figT, wspace=0.3, hspace=0.8)
            start = page_idx * rows_per_page
            end = min(start + rows_per_page, total_rows)
            for i, row in enumerate(summary_df.iloc[start:end].itertuples(), start=0):
                ax_img = figT.add_subplot(gsT[i, 0])
                ax_img.axis("off")
                code = row.MUTCD if pd.notnull(row.MUTCD) else "Unknown"
                img_path = os.path.join("mutcd_signs", f"{code}.png")
                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    trans = mtransforms.Affine2D().scale(0.5, 0.5)
                    ax_img.imshow(img, transform=trans + ax_img.transData)
                else:
                    ax_img.text(0.5, 0.5, f"No image\nfor\n{code}", ha="center", va="center", fontsize=7)
                texts = [
                    f"MUTCD: {code}",
                    f"Count: {row.Count}",
                    f"# With Text: {row._3}",
                    f"Avg Width: {row._4:.1f}",
                    f"Avg Height: {row._5:.1f}",
                    f"Avg Orient: {row._6:.1f}",
                    f"% of total: {row._7:.1f}%"
                ]
                for j in range(1, 8):
                    ax_txt = figT.add_subplot(gsT[i, j])
                    ax_txt.axis("off")
                    ax_txt.text(0.5, 0.5, texts[j - 1], ha="center", va="center", fontsize=7)
            figT.suptitle(f"Aggregated MUTCD Table - Page {page_idx + 1}/{num_pages}", fontsize=12)
            pdf.savefig(figT)
            plt.close(figT)
    print(f"Direct results report saved to {output_pdf}")


if __name__ == "__main__":
    pass
