# -*- coding: utf-8 -*-
"""
insert_screenshots.py
Inserts high-resolution screenshots of the live AOI web platform into Chapter 6
of PCBRisk_Final_Dissertation.docx and updates the List of Figures.
"""

import os
import sys
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def insert_figure(doc, anchor_elem, image_path, caption_text, width=Inches(6.0)):
    # Create image paragraph
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(3)
    run_img = p_img.add_run()
    run_img.add_picture(image_path, width=width)
    anchor_elem.addnext(p_img._p)
    
    # Create caption paragraph
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(12)
    run_cap = p_cap.add_run(caption_text)
    run_cap.font.name = 'Times New Roman'
    run_cap.font.size = Pt(10)
    run_cap.font.bold = True
    p_img._p.addnext(p_cap._p)
    
    return p_cap._p

def add_text_paragraph(doc, anchor_elem, text, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    anchor_elem.addnext(p._p)
    return p._p

def process_document(docx_path):
    print(f"Loading {docx_path}...")
    doc = docx.Document(docx_path)
    
    img_dir = r"c:\Users\priti\Downloads\pcbrisk\screenshots"
    img_overview = os.path.join(img_dir, "01_overview.png")
    img_workspace = os.path.join(img_dir, "02_workspace.png")
    img_results = os.path.join(img_dir, "03_results.png")
    img_evaluation = os.path.join(img_dir, "04_evaluation.png")
    img_history = os.path.join(img_dir, "05_history.png")
    
    # 1. Update List of Figures if needed
    for i, p in enumerate(doc.paragraphs):
        if "Figure 6.2: Live AOI Web Platform: Three-Queue Triage Interface" in p.text:
            print(f"Updating List of Figures around paragraph {i}...")
            anchor = p._p
            f63 = add_text_paragraph(doc, anchor, "Figure 6.3: Live AOI Web Platform: Plant Overview Dashboard ........................ [Page]", space_after=2)
            f64 = add_text_paragraph(doc, f63, "Figure 6.4: Live AOI Web Platform: Statistical Model Evaluation & Diagnostic Curves ........................ [Page]", space_after=2)
            f65 = add_text_paragraph(doc, f64, "Figure 6.5: Live AOI Web Platform: Historical Inspection Ledger & Forensic Audit Trail ........................ [Page]", space_after=2)
            break
            
    # 2. Locate Section 6.3 paragraphs and remove placeholder note
    p_board_inspector = None
    p_three_queue = None
    p_alpha_slider = None
    
    for i, p in enumerate(doc.paragraphs):
        if "[Note: Screenshots of the live web platform" in p.text:
            # Remove placeholder note
            p.text = ""
            p._p.getparent().remove(p._p)
            print("Removed placeholder note paragraph.")
            break

    # Re-scan after removal
    for i, p in enumerate(doc.paragraphs):
        if "In the Board Inspector View (Figure 6.1)" in p.text:
            p_board_inspector = p
        elif "In the Three-Queue Triage Interface (Figure 6.2)" in p.text:
            p_three_queue = p
        elif "Operators can interactively adjust the significance level alpha" in p.text:
            p_alpha_slider = p

    if not p_board_inspector or not p_three_queue:
        raise ValueError("Could not locate reference paragraphs in Section 6.3")

    # Insert Figure 6.1 right after p_board_inspector
    print("Inserting Figure 6.1 (Workspace / Board Inspector)...")
    cap_61 = insert_figure(
        doc,
        p_board_inspector._p,
        img_workspace,
        "Figure 6.1: Live AOI Web Platform — Board Inspector View (Inspection Workspace)"
    )

    # Insert Figure 6.2 right after p_three_queue
    print("Inserting Figure 6.2 (Three-Queue Triage / Results)...")
    cap_62 = insert_figure(
        doc,
        p_three_queue._p,
        img_results,
        "Figure 6.2: Live AOI Web Platform — Three-Queue Triage Interface & Results Dashboard"
    )

    # Insert Figures 6.3, 6.4, 6.5 after p_alpha_slider (or cap_62)
    last_anchor = cap_62 if p_alpha_slider is None else p_alpha_slider._p

    # Figure 6.3: Overview Dashboard
    text_63 = (
        "Figure 6.3 illustrates the plant-level Overview Dashboard (/overview), presenting real-time "
        "operational telemetry across active production shifts. Key indicators include cumulative PCB "
        "inspection throughput (128 boards), total defect findings (43 candidates), review queue volume "
        "(14 boards), first-pass manufacturing yield (89.1%), and a live activity stream of recently scanned panels."
    )
    p_desc_63 = add_text_paragraph(doc, last_anchor, text_63)
    cap_63 = insert_figure(
        doc,
        p_desc_63,
        img_overview,
        "Figure 6.3: Live AOI Web Platform — Plant Overview & Manufacturing Telemetry Dashboard"
    )

    # Figure 6.4: Model Evaluation
    text_64 = (
        "Figure 6.4 details the Model Evaluation & Diagnostic suite (/evaluation), providing statistical "
        "validation metrics for the fine-tuned YOLO11n defect detector. The interface displays overall "
        "precision (94.6%), recall (91.2%), F1 score (0.928), and mAP@50 (94.8%), accompanied by class-wise "
        "precision-recall curves, normalized confusion matrices, and conformal coverage reliability plots."
    )
    p_desc_64 = add_text_paragraph(doc, cap_63, text_64)
    cap_64 = insert_figure(
        doc,
        p_desc_64,
        img_evaluation,
        "Figure 6.4: Live AOI Web Platform — Statistical Model Evaluation & Diagnostic Telemetry"
    )

    # Figure 6.5: Inspection History
    text_65 = (
        "Figure 6.5 displays the Inspection History & Audit Trail interface (/history), delivering an immutable "
        "chronological ledger of all inspection events. Each entry logs batch serial numbers (e.g., PCB-2026-00128), "
        "board group classifications, defect severity counts, and operator disposition actions, ensuring full "
        "compliance with ISO 9001 and aerospace/automotive traceability standards."
    )
    p_desc_65 = add_text_paragraph(doc, cap_64, text_65)
    cap_65 = insert_figure(
        doc,
        p_desc_65,
        img_history,
        "Figure 6.5: Live AOI Web Platform — Historical Inspection Ledger & Forensic Audit Trail"
    )

    print(f"Saving updated document: {docx_path}...")
    doc.save(docx_path)
    print("Done!")

if __name__ == "__main__":
    process_document(r"C:\Users\priti\Downloads\pcbrisk\PCBRisk_Final_Dissertation.docx")
    process_document(r"C:\Users\priti\Downloads\pcbrisk\PCBRisk_Dissertation.docx")
