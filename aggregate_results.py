#!/usr/bin/env python3
"""
Combines every <run-key>.json written by train.py into one comparison table.

Run this any time -- after one job finishes or after all of them have --
since it just reads whatever JSON files currently exist in RESULTS_DIR.
This avoids the read-modify-write race that a single shared CSV would have
if multiple Slurm jobs finished around the same time and both tried to
append to it concurrently.

Usage:
    python aggregate_results.py
    python aggregate_results.py --results-dir /mnt/weka/etadevosyan/pcb-yolo/results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "PCB_RESULTS_DIR", "/mnt/weka/etadevosyan/pcb-yolo/results"
            )
        ),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Defaults to <results-dir>/comparison_table.csv",
    )
    args = p.parse_args()

    json_files = sorted(args.results_dir.glob("*.json"))
    if not json_files:
        print(f"No result JSON files found in {args.results_dir}")
        return

    rows = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        row = {k: v for k, v in data.items() if k != "per_class_ap50"}
        for cls, ap in data.get("per_class_ap50", {}).items():
            row[f"AP50_{cls.replace(' ', '_')}"] = ap
        rows.append(row)

    df = pd.DataFrame(rows)

    # Calculate F1 score and FEI (F1 Efficiency Index from Zhou & Agaian 2026: FEI = F1 * log10(T))
    if "precision" in df.columns and "recall" in df.columns:
        p = pd.to_numeric(df["precision"], errors="coerce")
        r = pd.to_numeric(df["recall"], errors="coerce")
        df["F1"] = (2 * p * r / (p + r + 1e-16)).fillna(0.0)

    if "F1" in df.columns and "fps" in df.columns:
        fps = pd.to_numeric(df["fps"], errors="coerce")
        df["FEI"] = (df["F1"] * np.log10(np.maximum(fps, 1.0))).fillna("")

    # 1. Define clean, intuitive column order
    priority_cols = [
        "model",
        "imgsz",
        "mAP50",
        "mAP50_95",
        "FEI",
        "F1",
        "precision",
        "recall",
        "fps",
        "total_time_ms",
        "AP50_Capacitor",
        "AP50_IC",
        "AP50_Connector",
        "AP50_Electrolytic_Capacitor",
        "epochs",
        "batch",
        "weights",
        "ensemble_models",
        "timestamp",
    ]

    # Keep priority columns that exist, followed by any remaining columns
    ordered_cols = [c for c in priority_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    # 2. Sort logically: highest mAP50 first (or by model)
    df = df.sort_values(by=["mAP50"], ascending=False).reset_index(drop=True)

    # 3. Round numeric metric columns to 4 decimal places for clean reading
    metric_cols = [
        "mAP50", "mAP50_95", "FEI", "F1", "precision", "recall",
        "AP50_Capacitor", "AP50_IC", "AP50_Connector", "AP50_Electrolytic_Capacitor"
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round(float(x), 4) if pd.notnull(x) and str(x).strip() != "" else "")

    if "fps" in df.columns:
        df["fps"] = df["fps"].apply(lambda x: round(float(x), 1) if pd.notnull(x) and str(x).strip() != "" else "")
    if "total_time_ms" in df.columns:
        df["total_time_ms"] = df["total_time_ms"].apply(lambda x: round(float(x), 1) if pd.notnull(x) and str(x).strip() != "" else "")
    if "imgsz" in df.columns:
        df["imgsz"] = df["imgsz"].apply(lambda x: int(x) if pd.notnull(x) and str(x).strip() != "" and not pd.isna(x) else "")
    if "epochs" in df.columns:
        df["epochs"] = df["epochs"].apply(lambda x: int(x) if pd.notnull(x) and str(x).strip() != "" and not pd.isna(x) else "")

    out_csv = args.out_csv or (args.results_dir / "comparison_table.csv")
    df.to_csv(out_csv, index=False)

    # 4. Also save a clean Markdown table for instant viewing (no external tabulate dependency)
    out_md = out_csv.with_suffix(".md")
    display_cols = [
        "model", "imgsz", "mAP50", "FEI", "F1", "fps", "total_time_ms", "AP50_Capacitor", "AP50_IC", "AP50_Connector"
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    
    with open(out_md, "w") as f:
        f.write("| " + " | ".join(display_cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(display_cols)) + " |\n")
        for _, r in df[display_cols].iterrows():
            f.write("| " + " | ".join(str(r[c]) for c in display_cols) + " |\n")

    # 5. Export styled Excel spreadsheet (.xlsx)
    out_xlsx = out_csv.with_suffix(".xlsx")
    export_excel(df, out_xlsx)

    print(f"Aggregated {len(df)} runs from {args.results_dir}")
    print(f"Saved CSV to:      {out_csv}")
    print(f"Saved Markdown to: {out_md}")
    print(f"Saved Excel to:    {out_xlsx}\n")


def export_excel(df: pd.DataFrame, out_path: Path):
    """Exports DataFrame to an Excel file with professional formatting."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Benchmark Results"

        # Styles
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
        zebra_fill = PatternFill(start_color="F7FAFC", end_color="F7FAFC", fill_type="solid")
        thin_border = Border(
            left=Side(style="thin", color="E2E8F0"),
            right=Side(style="thin", color="E2E8F0"),
            top=Side(style="thin", color="E2E8F0"),
            bottom=Side(style="thin", color="E2E8F0"),
        )

        # Write headers
        headers = list(df.columns)
        for col_idx, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_idx, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border
        ws.row_dimensions[1].height = 28

        # Write data rows
        for row_idx, (_, row) in enumerate(df.iterrows(), 2):
            is_zebra = (row_idx % 2 == 0)
            for col_idx, h in enumerate(headers, 1):
                val = row[h]
                cell = ws.cell(row=row_idx, column=col_idx, value=val)
                cell.border = thin_border
                if is_zebra:
                    cell.fill = zebra_fill
                
                # Center-align numbers and model names nicely
                if h in ["imgsz", "epochs", "batch", "fps", "total_time_ms"]:
                    cell.alignment = Alignment(horizontal="center")
                elif isinstance(val, (int, float)):
                    cell.alignment = Alignment(horizontal="right")
                else:
                    cell.alignment = Alignment(horizontal="left")
            ws.row_dimensions[row_idx].height = 20

        # Enable auto-filter on all columns
        ws.auto_filter.ref = ws.dimensions

        # Freeze the header row
        ws.freeze_panes = "A2"

        # Auto-fit column widths
        for col_idx, h in enumerate(headers, 1):
            col_letter = get_column_letter(col_idx)
            max_len = len(str(h))
            for cell in ws[col_letter]:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max_len + 4, 40)

        wb.save(out_path)
    except Exception as e:
        # Fallback to standard pandas to_excel
        df.to_excel(out_path, index=False)


if __name__ == "__main__":
    main()

