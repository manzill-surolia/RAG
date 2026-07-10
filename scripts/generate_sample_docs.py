#!/usr/bin/env python3
"""Generate sample PDF documents for testing the RAG system.

Consolidates the previous setup_pdf.py and generate_docs.py scripts.
Run:  python scripts/generate_sample_docs.py
"""

from __future__ import annotations

import os

from reportlab.pdfgen import canvas


DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "documents")


def _write_pdf(filepath: str, title: str, body: str) -> None:
    """Write a simple single- or multi-page PDF."""
    c = canvas.Canvas(filepath)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, title)
    c.setFont("Helvetica", 12)

    y = 750
    for line in body.split("\n"):
        if y < 50:
            c.showPage()
            y = 800
            c.setFont("Helvetica", 12)
        c.drawString(50, y, line.strip())
        y -= 20

    c.save()
    print(f"  Created {filepath}")


def main() -> None:
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    samples: list[tuple[str, str, str]] = [
        (
            "policy_general.pdf",
            "General ISMS Policy",
            """
1. Introduction
This document defines the Information Security Management System (ISMS).

2. Password Policy
Passwords must be at least 12 characters long and include special characters.
Passwords must be changed every 90 days.

3. Clean Desk Policy
All sensitive documents must be locked away when the employee leaves their desk.
""",
        ),
        (
            "regulation_gdpr.pdf",
            "GDPR Compliance Regulation",
            """
1. GDPR Compliance
The company adheres to the General Data Protection Regulation.

2. Data Subject Rights
Users have the right to request deletion of their data (Right to be Forgotten).
Requests must be processed within 30 days.

3. Breach Notification
In case of a data breach, the supervisory authority must be notified within 72 hours.
""",
        ),
        (
            "policy_data_retention.pdf",
            "Corporate Data Retention Policy",
            """
1. Purpose
This policy outlines the retention periods for company data to ensure compliance
with legal and business requirements.

2. General Retention Policy
All operational data shall be retained for a minimum of 5 years from the date of creation.
Financial records must be kept for 7 years.

3. User Data
User activity logs are retained for 1 year.
Personal Identifiable Information (PII) is retained only as long as the user account is active.

4. Deletion
Upon expiration of the retention period, data will be explicitly deleted
and overwritten to prevent recovery.
""",
        ),
    ]

    print(f"Generating {len(samples)} sample PDFs in '{DOCUMENTS_DIR}'...")
    for filename, title, body in samples:
        _write_pdf(os.path.join(DOCUMENTS_DIR, filename), title, body)
    print("Done.")


if __name__ == "__main__":
    main()
