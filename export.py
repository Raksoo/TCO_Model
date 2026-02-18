from __future__ import annotations

from io import BytesIO
from typing import Dict
from xml.sax.saxutils import escape
import zipfile

import pandas as pd


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def excel_col_name(idx: int) -> str:
    idx += 1
    name = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def _cell_xml(ref: str, value) -> str:
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float)) and not pd.isna(value):
        return f'<c r="{ref}"><v>{float(value)}</v></c>'
    txt = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{txt}</t></is></c>'


def _sheet_xml(df: pd.DataFrame) -> str:
    rows = []
    headers = list(df.columns)

    hdr_cells = []
    for c_idx, col in enumerate(headers):
        ref = f"{excel_col_name(c_idx)}1"
        hdr_cells.append(_cell_xml(ref, col))
    rows.append(f'<row r="1">{"".join(hdr_cells)}</row>')

    for r_idx, row in enumerate(df.itertuples(index=False), start=2):
        cells = []
        for c_idx, value in enumerate(row):
            ref = f"{excel_col_name(c_idx)}{r_idx}"
            cells.append(_cell_xml(ref, value))
        rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    dim_end_col = excel_col_name(max(0, len(headers) - 1))
    dim_end_row = max(1, len(df) + 1)

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{dim_end_col}{dim_end_row}"/>'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(rows)}</sheetData>'
        '</worksheet>'
    )


def _sanitize_sheet_name(name: str) -> str:
    bad = set('[]:*?/\\')
    out = "".join("_" if ch in bad else ch for ch in name)
    return out[:31] if out else "Sheet"


def dataframes_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Create a minimal XLSX file from pandas DataFrames without external engines."""
    bio = BytesIO()
    safe_names = [_sanitize_sheet_name(name) for name in sheets.keys()]
    sheet_items = list(sheets.items())

    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # [Content_Types]
        overrides = [
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        ]
        for i in range(1, len(sheet_items) + 1):
            overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )

        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'{"".join(overrides)}'
            '</Types>'
        )
        zf.writestr("[Content_Types].xml", content_types)

        # root rels
        root_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>'
        )
        zf.writestr("_rels/.rels", root_rels)

        # workbook + rels
        wb_sheets = []
        wb_rels = []
        for i, sheet_name in enumerate(safe_names, start=1):
            wb_sheets.append(
                f'<sheet name="{escape(sheet_name)}" sheetId="{i}" r:id="rId{i}"/>'
            )
            wb_rels.append(
                f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            )

        workbook = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{"".join(wb_sheets)}</sheets>'
            '</workbook>'
        )
        zf.writestr("xl/workbook.xml", workbook)

        workbook_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{"".join(wb_rels)}'
            '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '</Relationships>'
        )
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)

        styles = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
            '<cellXfs count="1"><xf xfId="0"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '</styleSheet>'
        )
        zf.writestr("xl/styles.xml", styles)

        for i, (_, df) in enumerate(sheet_items, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(df))

    return bio.getvalue()


def export_all(
    inputs_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> Dict[str, bytes]:
    sheets = {
        "Inputs": inputs_df,
        "Daily": daily_df,
        "Cashflow": cashflow_df,
        "Scenarios": scenarios_df,
        "Sensitivity": sensitivity_df,
    }

    return {
        "daily_csv": _to_csv_bytes(daily_df),
        "cashflow_csv": _to_csv_bytes(cashflow_df),
        "excel": dataframes_to_excel_bytes(sheets),
    }
