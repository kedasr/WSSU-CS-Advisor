"""
Shared constants and utility functions for the WSSU CS Graduate Advisor.

Used by both langchain_advisor.py (CLI) and streamlit_app.py (web).
"""

import re
from typing import List, Dict, Tuple

import pandas as pd
from pypdf import PdfReader

# Constants
MAX_CREDITS = 9
CHROMA_PERSIST_DIR = "./chroma_db"

# Degree Requirements
TRACK_REQUIREMENTS = {
    "thesis": {
        "core": {"CST 5320", "CST 5322", "CST 6306"},
        "required": {"CST 6301", "CST 6302"},
        "other": {"CST 6601"},
    },
    "project": {
        "core": set(),
        "required": {"CST 5325", "CST 5328", "CST 6305"},
        "other": {"CST 6312"},
    },
    "exam": {
        "core": set(),
        "required": {"CST 5320", "CST 6301", "CST 6302", "CST 5325", "CST 5328", "CST 6305"},
        "other": {"CST 6000"},
    }
}

SHARED_ELECTIVES = {
    "CST 5101", "CST 5130", "CST 5301", "CST 5302", "CST 5303", "CST 5304", "CST 5305",
    "CST 5306", "CST 5307", "CST 5308", "CST 5309", "CST 5316", "CST 5323", "CST 5324",
    "CST 5326", "CST 5329", "CST 5330", "CST 5331", "CST 5332", "CST 5333", "CST 5334",
    "CST 5335", "CST 5340", "CST 5350", "CST 6000", "CST 6130", "CST 6303", "CST 6304",
    "CST 6307", "CST 6308", "CST 6309", "CST 6310", "CST 6311", "CST 6314", "CST 6320",
    "CST 7130"
}


def extract_courses_from_text(text: str) -> set[str]:
    """Extract CST course codes from text content."""
    return set(re.findall(r"\bCST\s+\d{4}\b", text))


def extract_taken_courses_from_pdf(pdf_source) -> set[str]:
    """Extract completed course codes from a transcript PDF.

    Args:
        pdf_source: Either a file path (str) or a file-like object.
    """
    reader = PdfReader(pdf_source)
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    return extract_courses_from_text(text)


def load_spring_courses(excel_path: str = "data/Spring2026_Courses.xlsx") -> pd.DataFrame:
    """Load and normalize Spring 2026 course offerings."""
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    df["Title"] = df["Title"].astype(str).str.strip()
    df["Subject"] = df["Subject"].astype(str).str.strip()
    df["Course Number"] = df["Course Number"].astype(str).str.strip()
    df["code"] = df["Subject"] + " " + df["Course Number"]
    df["Credits"] = pd.to_numeric(df["Credits"], errors="coerce").fillna(3).astype(int)
    df["Days"] = df["Meeting Days"].astype(str).str.strip()
    df["Time"] = df["Meeting Times"].astype(str).str.strip()

    if "Instructional Menthods." in df.columns:
        df["Instruction Mode"] = df["Instructional Menthods."].astype(str).str.strip()
    else:
        df["Instruction Mode"] = ""

    df["Status"] = df["Status"].astype(str).str.strip() if "Status" in df.columns else ""
    return df


def is_online_or_async(mode: str) -> bool:
    """Check if a course is online or asynchronous."""
    m = (mode or "").lower()
    return "online" in m or "async" in m or "asynchronous" in m


def time_conflicts(course_row: dict, selected_rows: list[dict]) -> bool:
    """Check whether a course conflicts with already-selected courses."""
    if is_online_or_async(course_row.get("Instruction Mode", "")):
        return False

    c_days = str(course_row.get("Days", "")).strip()
    c_time = str(course_row.get("Time", "")).strip()

    if not c_days or not c_time or c_days.lower() == "nan" or c_time.lower() == "nan":
        return False

    for s in selected_rows:
        if is_online_or_async(s.get("Instruction Mode", "")):
            continue
        if c_days == s.get("Days") and c_time == s.get("Time"):
            return True

    return False


def pick_courses_for_track(
    track: str, taken: set[str], spring_df: pd.DataFrame
) -> Tuple[List[dict], int]:
    """Select courses based on track and completed coursework."""
    req = TRACK_REQUIREMENTS[track]
    needed = (req["core"] | req["required"] | req["other"]) - taken

    spring_df = spring_df[spring_df["Subject"].str.upper() == "CST"].copy()

    def status_rank(x: str) -> int:
        x = (x or "").lower()
        if "open" in x:
            return 0
        if "wait" in x:
            return 1
        return 2

    spring_df["status_rank"] = spring_df["Status"].apply(status_rank)
    spring_df = spring_df.sort_values(["status_rank", "code"])

    selected: List[dict] = []
    total_credits = 0

    def try_add(row: dict) -> bool:
        nonlocal total_credits
        cr = row.get("Credits", 3)
        if total_credits + cr > MAX_CREDITS:
            return False
        if time_conflicts(row, selected):
            return False
        selected.append(row)
        total_credits += cr
        return True

    for _, row in spring_df[spring_df["code"].isin(needed)].iterrows():
        if total_credits >= MAX_CREDITS:
            break
        try_add(row.to_dict())

    if total_credits < MAX_CREDITS:
        electives = spring_df[
            spring_df["code"].isin(SHARED_ELECTIVES)
            & (~spring_df["code"].isin(taken))
            & (~spring_df["code"].isin([s["code"] for s in selected]))
        ]
        for _, row in electives.iterrows():
            if total_credits >= MAX_CREDITS:
                break
            try_add(row.to_dict())

    return selected, total_credits
