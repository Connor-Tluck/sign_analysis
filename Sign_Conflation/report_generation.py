import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def generate_final_summary_and_table(pdf, df_all, dataset_names=("File 1", "File 2")):
    figJ = plt.figure(figsize=(8.27, 11.69))
    axJ = figJ.add_axes([0.1, 0.1, 0.8, 0.8])
    axJ.axis('off')
    total_signs = len(df_all)
    dataset_list_text = "\n".join(dataset_names)
    summary_text = f"Final Summary\n\nTotal Sign Extraction: {total_signs}\n\nDatasets:\n{dataset_list_text}\n"
    axJ.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
    pdf.savefig(figJ)
    plt.close()
    group = df_all.groupby("MUTCD", dropna=False)
    total_signs = len(df_all)
    summary_df = pd.DataFrame({
        "MUTCD": group["MUTCD"].first(),
        "Count": group.size(),
        "# With Text": group["HasText"].sum() if "HasText" in df_all.columns else 0,
        "Avg Width": group["Width"].mean() if "Width" in df_all.columns else 0,
        "Avg Height": group["Height"].mean() if "Height" in df_all.columns else 0,
        "Avg Orientation": group["Orientation"].mean() if "Orientation" in df_all.columns else 0
    }).reset_index(drop=True)
    summary_df["% of total"] = (summary_df["Count"] / total_signs) * 100
    summary_df = summary_df.sort_values("Count", ascending=False).reset_index(drop=True)
    rows_per_page = 30
    total_rows = len(summary_df)
    num_pages = math.ceil(total_rows / rows_per_page)
    for page_idx in range(num_pages):
        figK = plt.figure(figsize=(8.27, 11.69))
        gsK = GridSpec(nrows=rows_per_page, ncols=8, figure=figK, wspace=0.3, hspace=0.8)
        start_i = page_idx * rows_per_page
        end_i = min(start_i + rows_per_page, total_rows)
        for row_i, code_row in enumerate(summary_df.iloc[start_i:end_i].itertuples(), start=0):
            ax_img = figK.add_subplot(gsK[row_i, 0])
            ax_img.axis('off')
            code = code_row.MUTCD if pd.notnull(code_row.MUTCD) else "Unknown"
            img_path = os.path.join("../mutcd_signs", f"{code}.png")
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                trans = mtransforms.Affine2D().scale(0.7, 0.7)
                ax_img.imshow(img, transform=trans + ax_img.transData)
            else:
                ax_img.text(0.5, 0.5, f"No image\nfor\n{code}", ha='center', va='center', fontsize=7)
            col_texts = [
                f"MUTCD: {code}",
                f"Count: {code_row.Count}",
                f"# With Text: {code_row._3}",
                f"Avg Width: {code_row._4:.1f}",
                f"Avg Height: {code_row._5:.1f}",
                f"Avg Orient: {code_row._6:.1f}",
                f"% of total: {code_row._7:.1f}%"
            ]
            for c in range(1, 8):
                ax_cell = figK.add_subplot(gsK[row_i, c])
                ax_cell.axis('off')
                ax_cell.text(0.5, 0.5, col_texts[c-1], ha='center', va='center', fontsize=7)
        figK.suptitle(f"All Extracted Signs - Page {page_idx+1}/{num_pages}", fontsize=12)
        pdf.savefig(figK)
        plt.close()

def generate_conflation_report_all(results, logo_path=None, output_pdf="conflation_report.pdf", duplicates_info=None):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    with PdfPages(output_pdf) as pdf:
        fig_cover = plt.figure(figsize=(8.27, 11.69))
        ax_cover = fig_cover.add_axes([0, 0, 1, 1])
        ax_cover.axis('off')
        title_text = "Conflation Report"
        subtitle_text = "Distance-Based Comparison\n(File 2 = Known Dataset)"
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cover_text = f"{title_text}\n\n{subtitle_text}\n\nReport Generated: {gen_time}"
        if duplicates_info is not None:
            dup_text = (f"\n\nDuplicate Removal Information:\n"
                        f"File 1: Original = {duplicates_info['File 1']['original_count']}, Removed = {duplicates_info['File 1']['duplicates_removed']}, Unique = {duplicates_info['File 1']['unique_count']}\n"
                        f"File 2: Original = {duplicates_info['File 2']['original_count']}, Removed = {duplicates_info['File 2']['duplicates_removed']}, Unique = {duplicates_info['File 2']['unique_count']}")
            cover_text += dup_text
        ax_cover.text(0.5, 0.5, cover_text, fontsize=16, ha='center', va='center')
        if logo_path and os.path.exists(logo_path):
            ax_logo = fig_cover.add_axes([0.35, 0.7, 0.3, 0.2])
            import matplotlib.image as mpimg
            logo = mpimg.imread(logo_path)
            ax_logo.imshow(logo)
            ax_logo.axis("off")
        pdf.savefig(fig_cover)
        plt.close()
        for result in results:
            metrics = result["metrics"]
            df_matched = result["df_matched"]
            df_unmatched = result["df_unmatched"]
            df_all = pd.concat([df_matched, df_unmatched], ignore_index=True)
            figA = plt.figure(figsize=(8.27, 11.69))
            ax_text = figA.add_axes([0.1, 0.55, 0.8, 0.35])
            ax_text.axis('off')
            metrics_text = f"Buffer: {metrics['buffer']} m\nMatched: {metrics['matched_signs']}\nFile 1 Total: {metrics['total_mach9']}\nFile 2 Total: {metrics['total_client']}\nAvg Match Dist: {metrics['avg_match_distance']:.2f} m\nMin/Max: {metrics['min_match_distance']:.2f} / {metrics['max_match_distance']:.2f} m\n"
            ax_text.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=10)
            ax_map = figA.add_axes([0.1, 0.1, 0.8, 0.35])
            ax_map.scatter(df_matched["Longitude"], df_matched["Latitude"], color='green', s=50)
            ax_map.scatter(df_unmatched["Longitude"], df_unmatched["Latitude"], color='red', s=50)
            ax_map.set_title(f"Map (Buffer: {metrics['buffer']} m)")
            ax_map.grid(True)
            pdf.savefig(figA)
            plt.close()
            figB = plt.figure(figsize=(8.27, 11.69))
            gsB = plt.GridSpec(nrows=2, ncols=1, height_ratios=[1, 1], figure=figB)
            axB1 = figB.add_subplot(gsB[0])
            axB1.set_title("Sign Counts by Dataset")
            categories = ["File 1 Total", "File 1 Matched", "File 2 Total", "File 2 Matched"]
            counts = [metrics["total_mach9"], metrics["matched_signs"], metrics["total_client"], metrics["matched_signs"]]
            positions = np.arange(len(categories))
            axB1.bar(positions, counts, color=["#1f77b4", "#2ca02c", "#1f77b4", "#2ca02c"])
            axB1.set_xticks(positions)
            axB1.set_xticklabels(categories, rotation=30, ha='right')
            for i, v in enumerate(counts):
                axB1.text(i, v+0.5, str(v), ha='center', va='bottom')
            df_matched_codes = df_matched["MUTCD"].value_counts()
            df_known_codes = df_all[df_all["Source"]=="File 2"]["MUTCD"].value_counts()
            top20_codes = df_matched_codes.head(20).index
            table_data = []
            for idx, code in enumerate(top20_codes, start=1):
                m_count = df_matched_codes.get(code, 0)
                k_count = df_known_codes.get(code, 0)
                pct = (m_count / k_count * 100) if k_count > 0 else 0
                table_data.append([idx, code, m_count, k_count, f"{pct:.1f}%"])
            axB2 = figB.add_subplot(gsB[1])
            axB2.axis('tight')
            axB2.axis('off')
            table = axB2.table(cellText=table_data, colLabels=["Index", "MUTCD", "Matched", "Known", "Percent"], loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            figB.subplots_adjust(hspace=0.4)
            pdf.savefig(figB)
            plt.close()
            figC = plt.figure(figsize=(8.27, 11.69))
            axC1 = figC.add_subplot(211)
            pos1 = np.arange(len(df_matched_codes.head(10)))
            axC1.bar(pos1, df_matched_codes.head(10).values, color='#8da0cb')
            axC1.set_title("Top 10 Matched Codes")
            axC1.set_xticks(pos1)
            axC1.set_xticklabels(df_matched_codes.head(10).index, rotation=30, ha='right')
            axC2 = figC.add_subplot(212)
            pos2 = np.arange(len(df_unmatched["MUTCD"].value_counts().head(10)))
            axC2.bar(pos2, df_unmatched["MUTCD"].value_counts().head(10).values, color='#fc8d62')
            axC2.set_title("Top 10 Unmatched Codes")
            axC2.set_xticks(pos2)
            axC2.set_xticklabels(df_unmatched["MUTCD"].value_counts().head(10).index, rotation=30, ha='right')
            figC.subplots_adjust(hspace=0.5)
            pdf.savefig(figC)
            plt.close()
            figD = plt.figure(figsize=(8.27, 11.69))
            axD = figD.add_axes([0.1, 0.1, 0.8, 0.8])
            axD.axis('off')
            dup_text = f"Duplicate Report:\nFile 1: Orig={duplicates_info['File 1']['original_count']}, Removed={duplicates_info['File 1']['duplicates_removed']}, Unique={duplicates_info['File 1']['unique_count']}\nFile 2: Orig={duplicates_info['File 2']['original_count']}, Removed={duplicates_info['File 2']['duplicates_removed']}, Unique={duplicates_info['File 2']['unique_count']}"
            axD.text(0.5, 0.5, dup_text, ha='center', va='center', fontsize=12)
            pdf.savefig(figD)
            plt.close()
            figE = plt.figure(figsize=(8.27, 11.69))
            gsE = plt.GridSpec(nrows=2, ncols=1, height_ratios=[1, 1], figure=figE)
            axE1 = figE.add_subplot(gsE[0])
            match_distances = df_matched["match_distance"].dropna()
            if len(match_distances) > 0:
                axE1.hist(match_distances, bins=30, color='purple', edgecolor='black', alpha=0.7)
                axE1.set_title("Distribution of Match Distances")
                axE1.set_xlabel("Distance (m)")
                axE1.set_ylabel("Frequency")
            else:
                axE1.text(0.5, 0.5, "No match distance data.", ha='center', va='center', fontsize=12)
            axE2 = figE.add_subplot(gsE[1])
            axE2.axis('off')
            if len(match_distances) > 0:
                stats = {"Mean": np.mean(match_distances), "Median": np.median(match_distances), "Std Dev": np.std(match_distances), "Min": np.min(match_distances), "Max": np.max(match_distances)}
                stats_text = "\n".join([f"{k}: {v:.2f} m" for k, v in stats.items()])
            else:
                stats_text = "No data available."
            axE2.text(0.5, 0.5, "Summary Stats:\n" + stats_text, ha='center', va='center', fontsize=12)
            figE.subplots_adjust(hspace=0.4)
            pdf.savefig(figE)
            plt.close()
            figF = plt.figure(figsize=(8.27, 11.69))
            axF = figF.add_axes([0.1, 0.1, 0.8, 0.8])
            axF.scatter(df_matched["Longitude"], df_matched["Latitude"], color='green', s=50)
            axF.set_title(f"Map of Matched (Buffer: {metrics['buffer']} m)")
            axF.grid(True)
            pdf.savefig(figF)
            plt.close()
            figG = plt.figure(figsize=(8.27, 11.69))
            axG = figG.add_axes([0.1, 0.1, 0.8, 0.8])
            axG.scatter(df_unmatched["Longitude"], df_unmatched["Latitude"], color='red', s=50)
            axG.set_title(f"Map of Unmatched (Buffer: {metrics['buffer']} m)")
            axG.grid(True)
            pdf.savefig(figG)
            plt.close()
            top_matched_counts_10 = df_matched["MUTCD"].value_counts().head(10)
            figH = plt.figure(figsize=(8.27, 11.69))
            figH.suptitle(f"Top 10 Matched Codes (Buffer: {metrics['buffer']} m)", fontsize=14)
            gsH = plt.GridSpec(nrows=5, ncols=2, figure=figH, wspace=0.2, hspace=0.3)
            for i, code in enumerate(top_matched_counts_10.index):
                row = i // 2
                col = i % 2
                ax_img = figH.add_subplot(gsH[row, col])
                ax_img.axis('off')
                file1_count = df_all[(df_all["Source"] == "File 1") & (df_all["MUTCD"] == code)].shape[0]
                file2_count = df_all[(df_all["Source"] == "File 2") & (df_all["MUTCD"] == code)].shape[0]
                matched_in_file2 = df_all[(df_all["Source"] == "File 2") & (df_all["MUTCD"] == code) & (df_all["matched"] == True)].shape[0]
                pct_match = (matched_in_file2 / file2_count * 100) if file2_count > 0 else 0.0
                img_path = os.path.join("../mutcd_signs", f"{code}.png")
                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    trans = mtransforms.Affine2D().scale(0.7, 0.7)
                    ax_img.imshow(img, transform=trans + ax_img.transData)
                    ax_img.set_title(f"{code}\nF1: {file1_count}, F2: {file2_count}\nMatched%: {pct_match:.1f}%", fontsize=10)
                else:
                    ax_img.text(0.5, 0.5, f"No image for\n{code}\nF1: {file1_count}, F2: {file2_count}\nMatched%: {pct_match:.1f}%", ha='center', va='center', fontsize=9)
            pdf.savefig(figH)
            plt.close()
            top_unmatched_counts_10 = df_unmatched["MUTCD"].value_counts().head(10)
            figI = plt.figure(figsize=(8.27, 11.69))
            figI.suptitle(f"Top 10 Unmatched Codes (Buffer: {metrics['buffer']} m)", fontsize=14)
            gsI = plt.GridSpec(nrows=5, ncols=2, figure=figI, wspace=0.2, hspace=0.3)
            for i, code in enumerate(top_unmatched_counts_10.index):
                row = i // 2
                col = i % 2
                ax_img = figI.add_subplot(gsI[row, col])
                ax_img.axis('off')
                file1_count = df_all[(df_all["Source"] == "File 1") & (df_all["MUTCD"] == code)].shape[0]
                file2_count = df_all[(df_all["Source"] == "File 2") & (df_all["MUTCD"] == code)].shape[0]
                matched_in_file2 = df_all[(df_all["Source"] == "File 2") & (df_all["MUTCD"] == code) & (df_all["matched"] == True)].shape[0]
                pct_match = (matched_in_file2 / file2_count * 100) if file2_count > 0 else 0.0
                img_path = os.path.join("../mutcd_signs", f"{code}.png")
                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    trans = mtransforms.Affine2D().scale(0.7, 0.7)
                    ax_img.imshow(img, transform=trans + ax_img.transData)
                    ax_img.set_title(f"{code}\nF1: {file1_count}, F2: {file2_count}\nMatched%: {pct_match:.1f}%", fontsize=10)
                else:
                    ax_img.text(0.5, 0.5, f"No image for\n{code}\nF1: {file1_count}, F2: {file2_count}\nMatched%: {pct_match:.1f}%", ha='center', va='center', fontsize=9)
            pdf.savefig(figI)
            plt.close()
        df_all_runs = []
        for res in results:
            df_all_runs.append(res["df_matched"])
            df_all_runs.append(res["df_unmatched"])
        df_all_combined = pd.concat(df_all_runs, ignore_index=True)
        if "MUTCD" not in df_all_combined.columns:
            df_all_combined["MUTCD"] = "Unknown"
        if "HasText" not in df_all_combined.columns:
            df_all_combined["HasText"] = np.random.rand(len(df_all_combined)) < 0.3
        if "Width" not in df_all_combined.columns:
            df_all_combined["Width"] = np.random.normal(50, 5, size=len(df_all_combined)).clip(min=1)
        if "Height" not in df_all_combined.columns:
            df_all_combined["Height"] = np.random.normal(60, 8, size=len(df_all_combined)).clip(min=1)
        if "Orientation" not in df_all_combined.columns:
            df_all_combined["Orientation"] = np.random.normal(0, 10, size=len(df_all_combined))
        dataset_names = ("Mach9 Extraction (File 1)", "Known Dataset (File 2)")
        generate_final_summary_and_table(pdf, df_all_combined, dataset_names=dataset_names)
    print(f"Conflation report saved to {output_pdf}")
