"""
generate_sample_docs.py — Enterprise Knowledge Base Document Generator
========================================================================
Generates detailed, realistic corporate policy documents across 4 formats:
  • PDF  : Global Business Travel & Expense Policy
  • DOCX : Employee Health, Wellness & Parental Leave Benefits
  • MD   : IT Infrastructure, Hardware Stipend & Cyber Security
  • TXT  : Enterprise Remote Work & Hybrid Operations Policy
"""

import os
import docx
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_all_docs(docs_dir: str = "documents"):
    """Generates all sample corporate policy documents."""
    os.makedirs(docs_dir, exist_ok=True)

    # 1. Generate TXT Document — Enterprise Remote Work Policy
    txt_path = os.path.join(docs_dir, "remote_work_policy.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            "[Doc 201] Enterprise Remote Work and Hybrid Operations Policy\n"
            "Title: Hybrid Work Guidelines and Remote Eligibility\n"
            "Effective Date: January 1, 2026\n"
            "Scope: All Global Full-Time Employees\n\n"
            "SECTION 1: ELIGIBILITY & APPROVAL WORKFLOW\n"
            "All full-time employees who have completed at least 90 calendar days of continuous "
            "service are eligible to request hybrid or full remote work arrangements. Requests must be "
            "submitted through the HR Portal and approved by the VP of Department Operations.\n\n"
            "SECTION 2: STIPENDS & HOME OFFICE SETUP\n"
            "1. Home Internet Stipend: A recurring stipend of $85 per month is paid automatically on the "
            "first pay period of each month to cover broadband internet costs.\n"
            "2. Office Furniture Credit: Remote team members receive a one-time reimbursement of up to $600 "
            "for ergonomic desk equipment (chairs, standing desks, monitor arms).\n"
            "3. Core Hours Requirement: All remote staff must remain available on Slack and Zoom during "
            "core business hours (10:00 AM to 4:00 PM EST).\n\n"
            "SECTION 3: INTERNATIONAL REMOTE WORK (WORKATION)\n"
            "Employees may work remotely outside their primary residence state/country for up to 30 calendar days "
            "per fiscal year, subject to prior tax compliance review by the legal department at least 14 days prior to departure.\n"
        )

    # 2. Generate Markdown Document — IT Equipment & Cyber Security Policy
    md_path = os.path.join(docs_dir, "it_security_and_equipment.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            "# [Doc 202] IT Equipment, Laptop Hardware & Cyber Security Policy\n"
            "**Document Version:** 4.2 | **Owner:** Information Security Team\n\n"
            "## Section 1: Standard Hardware Provisioning\n"
            "Upon joining the organization, every software engineer, product manager, and designer is issued hardware "
            "based on their role tier:\n\n"
            "- **Engineering & Architecture:** Apple MacBook Pro 16\" (M3 Max, 36GB RAM) or Dell Precision Workstation (64GB RAM).\n"
            "- **General Operations:** Apple MacBook Air 15\" (M3, 16GB RAM) or Lenovo ThinkPad X1 Carbon.\n"
            "- **Refresh Cycle:** Laptops are refreshed every 36 months (3 years). Upon completion of 3 years, employees may choose to keep their legacy machine for a nominal $50 fee or turn it in for safe recycling.\n\n"
            "## Section 2: Security & Encryption Protocols\n"
            "1. **Full-Disk Encryption:** FileVault (macOS) or BitLocker (Windows) must remain enabled at all times with strict recovery keys managed by Okta MDM.\n"
            "2. **Multi-Factor Authentication (MFA):** Hardware security keys (YubiKey) are required for all access to AWS, GitHub, and production Kubernetes clusters.\n"
            "3. **Password Vaulting:** All corporate logins must use 1Password. Plaintext passwords stored in local files, Slack messages, or Git repos are flagged automatically and trigger a security escalation.\n"
            "4. **VPN Requirements:** All public network connections (airports, cafes, hotels) must connect via Cloudflare Zero Trust VPN prior to accessing internal administrative APIs.\n"
        )

    # 3. Generate DOCX Document — Comprehensive Employee Benefits
    docx_path = os.path.join(docs_dir, "employee_benefits_guide.docx")
    doc = docx.Document()

    doc.add_heading('[Doc 203] Global Employee Health, Wellness & Family Care Benefits', level=1)
    doc.add_paragraph('Document Ref: HR-BEN-2026-V1 | Applicable Region: North America & EMEA')

    p1 = doc.add_paragraph()
    p1.add_run('1. Comprehensive Healthcare Coverage\n').bold = True
    p1.add_run(
        'The company offers 100% employer-paid health, dental, and vision insurance premiums for all primary employees, '
        'and 80% coverage for eligible dependents. Employees can choose between a High-Deductible Health Plan (HDHP) '
        'with a company-funded Health Savings Account (HSA) contribution of $1,500/year, or a traditional Preferred Provider Organization (PPO) plan.'
    )

    p2 = doc.add_paragraph()
    p2.add_run('\n2. Annual Wellness & Mental Health Stipend\n').bold = True
    p2.add_run(
        'Every full-time team member receives an annual wellness allowance of $750. Eligible expenses include gym memberships, '
        'fitness trackers (e.g., Apple Watch, Oura Ring), mental health apps (Calm, Headspace), ergonomic office chairs, and sports equipment. '
        'In addition, 12 confidential therapy sessions per year are provided completely free through Lyra Health.'
    )

    p3 = doc.add_paragraph()
    p3.add_run('\n3. Parental Leave & Family Building\n').bold = True
    p3.add_run(
        'All primary caregivers receive 18 weeks of fully paid parental leave for birth, adoption, or foster placement. '
        'Secondary caregivers receive 10 weeks of fully paid leave. Furthermore, the company provides up to $10,000 in lifetime '
        'reimbursement for fertility treatments, IVF, or adoption agency fees.'
    )

    doc.save(docx_path)

    # 4. Generate PDF Document — Corporate Travel & Expense Policy
    pdf_path = os.path.join(docs_dir, "corporate_travel_policy.pdf")
    doc_pdf = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    story = []

    # Title & Metadata
    story.append(Paragraph("<b>[Doc 204] Corporate Travel & Expense Policy (FY2026)</b>", styles['Title']))
    story.append(Spacer(1, 10))

    intro_text = (
        "This policy outlines mandatory guidelines for official corporate travel, client entertainments, and business "
        "reimbursements. All travel bookings must be made through Navan (formerly TripActions) at least 14 days prior to domestic travel "
        "and 21 days prior to international travel."
    )
    story.append(Paragraph(intro_text, styles['Normal']))
    story.append(Spacer(1, 15))

    # Section Header
    story.append(Paragraph("<b>Per Diem & Expense Allowances Table</b>", styles['Heading2']))
    story.append(Spacer(1, 10))

    # Table Data
    table_data = [
        ["Category", "Domestic Travel Allowance", "International Travel Allowance", "Receipt Requirement"],
        ["Daily Meal Per Diem", "$85 / day", "$120 / day", "Required if single transaction > $25"],
        ["Hotel Lodging Limit", "$250 / night (Tier 1 cities)", "$350 / night (Tier 1 cities)", "Mandatory itemized folio"],
        ["Rideshare / Taxi", "Uber Comfort / Lyft Preferred", "Licensed taxis / Executive sedan", "Required for all fares"],
        ["Flight Class", "Economy (Under 6 hours)", "Business Class (Over 8 hours)", "Boarding pass required"]
    ]

    t = Table(table_data, colWidths=[120, 150, 150, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
    ]))

    story.append(t)
    story.append(Spacer(1, 20))

    story.append(Paragraph("<b>Policy Violation Exceptions & Non-Reimbursable Items</b>", styles['Heading3']))
    story.append(Paragraph(
        "Alcoholic beverages consumed outside of group business dinners, personal entertainment expenses (in-room movies, mini-bar), "
        "traffic violations, and flight upgrade charges without VP approval will not be reimbursed under any circumstances.",
        styles['Normal']
    ))

    doc_pdf.build(story)

    print("✅ Successfully generated detailed, enterprise-grade documents in documents/ directory:")
    print(f" - {txt_path}")
    print(f" - {md_path}")
    print(f" - {docx_path}")
    print(f" - {pdf_path}")


if __name__ == "__main__":
    generate_all_docs()