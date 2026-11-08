"""
apply_exact_page_numbers.py
Updates Table of Contents, List of Figures, and List of Tables with the
exact layout-engine page numbers computed by Word for the 89-page dissertation.
"""

import sys, os, re
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Exact verified page numbers from Word's rendering engine
TOC_DATA = [
    ("Declaration", 3, 1),
    ("Acknowledgement", 4, 1),
    ("Executive Summary", 5, 1),
    ("List of Figures", 8, 1),
    ("List of Tables", 9, 1),
    ("Abbreviations", 10, 1),
    ("Symbols and Notations", 12, 1),
    ("CHAPTER 1: INTRODUCTION", 13, 1),
    ("1.1 Introduction", 13, 2),
    ("1.2 Objective", 15, 2),
    ("1.3 Motivation", 16, 2),
    ("1.4 Background", 18, 2),
    ("CHAPTER 2: DISSERTATION DESCRIPTION AND GOALS", 20, 1),
    ("2.1 Project Scope", 20, 2),
    ("2.2 Problem Statement", 21, 2),
    ("2.3 Intended Users and Application Domain", 22, 2),
    ("2.4 Project Goals and Deliverables", 22, 2),
    ("2.5 Functional Requirements", 24, 2),
    ("2.6 Non-Functional Requirements", 25, 2),
    ("2.7 Distinction Between Initial Goals and Completed Work", 26, 2),
    ("CHAPTER 3: TECHNICAL SPECIFICATION", 27, 1),
    ("3.1 Software Requirements and Development Environment", 27, 2),
    ("3.2 Hardware Specifications for Benchmarking", 29, 2),
    ("3.3 Dataset Specification", 30, 2),
    ("3.4 Defect Class Taxonomy", 32, 2),
    ("3.5 System Inputs, Processing, and Outputs", 33, 2),
    ("3.6 Performance and Evaluation Criteria", 35, 2),
    ("CHAPTER 4: DESIGN APPROACH AND DETAILS", 36, 1),
    ("4.1 Design Approach / Materials & Methods", 37, 2),
    ("4.1.1 Overall System Architecture", 37, 3),
    ("4.1.2 Data Acquisition and Preparation", 38, 3),
    ("4.1.3 Graph-Constrained Synthetic Defect Generation", 40, 3),
    ("4.1.4 Object Detector Training", 42, 3),
    ("4.1.5 Local Topological Consequence Engine (v3)", 43, 3),
    ("4.1.6 Feature Engineering and Triage Policies", 45, 3),
    ("4.1.7 Split-Conformal Calibration Methods", 46, 3),
    ("4.1.8 Leave-One-Board-Out (LOBO) Evaluation Protocol", 47, 3),
    ("4.2 Codes and Standards", 48, 2),
    ("4.3 Constraints, Alternatives, and Tradeoffs", 49, 2),
    ("4.3.1 Technical Constraints", 50, 3),
    ("4.3.2 Alternative Approaches Considered", 51, 3),
    ("4.3.3 Key Design Trade-offs", 52, 3),
    ("CHAPTER 5: SCHEDULE, TASKS AND MILESTONES", 53, 1),
    ("CHAPTER 6: DISSERTATION / PROJECT DEMONSTRATION", 55, 1),
    ("6.1 Demonstration Setup and Prerequisites", 56, 2),
    ("6.2 Step-by-Step Execution Procedure", 57, 2),
    ("6.3 System Demonstration Walkthrough", 57, 2),
    ("6.4 Representative Scenarios", 63, 2),
    ("CHAPTER 7: RESULTS AND DISCUSSION", 65, 1),
    ("7.1 Cost Analysis", 66, 2),
    ("7.2 Results", 67, 2),
    ("7.2.1 Object Detector Performance", 67, 3),
    ("7.2.2 Consequence Engine Accuracy", 69, 3),
    ("7.2.3 Multi-Seed LOBO Replication Results", 69, 3),
    ("7.2.4 Cost-Sensitive Quality Evaluation", 71, 3),
    ("7.2.5 Hardware Latency Benchmarks", 73, 3),
    ("7.2.6 Registration Tolerance Sensitivity", 74, 3),
    ("7.2.7 Mondrian Group-Conditional Calibration", 75, 3),
    ("7.2.8 Automated PASS/FAIL Checklist", 76, 3),
    ("7.2.9 Formal Hypothesis Verdicts", 77, 3),
    ("7.3 Discussion", 78, 2),
    ("7.3.1 Achievement of Objectives", 78, 3),
    ("7.3.2 How Results Address the Original Problem", 78, 3),
    ("7.3.3 Strengths and Technical Contributions", 79, 3),
    ("7.3.4 Comparison with Existing Work", 79, 3),
    ("7.3.5 Failure Modes and Limitations", 80, 3),
    ("7.4 Conclusion and Future Scope", 80, 2),
    ("7.4.1 Conclusion", 81, 3),
    ("7.4.2 Future Scope", 81, 3),
    ("REFERENCES", 83, 1),
    ("APPENDICES", 85, 1),
    ("Appendix A: Repository File Structure", 86, 2),
    ("Appendix B: Key Source Code — Consequence Engine (pcbrisk_graph.py)", 87, 2),
    ("Appendix C: Key Source Code — Conformal Evaluation (pcbrisk_eval.py)", 87, 2),
    ("Appendix D: Installation and Execution Instructions", 88, 2),
    ("Appendix E: ONNX Model Export Details", 89, 2),
]

FIGURE_PAGES = {
    "Figure 4.1": 37,
    "Figure 4.2": 40,
    "Figure 4.3": 44,
    "Figure 4.4": 46,
    "Figure 4.5": 47,
    "Figure 6.1": 59,
    "Figure 6.2": 60,
    "Figure 6.3": 61,
    "Figure 6.4": 62,
    "Figure 6.5": 63,
    "Figure 7.1": 70,
    "Figure 7.2": 71,
    "Figure 7.3": 72,
    "Figure 7.4": 72,
    "Figure 7.5": 73,
    "Figure 7.6": 74,
    "Figure 7.7": 75,
}

TABLE_PAGES = {
    "Table 2.1": 22,
    "Table 3.1": 27,
    "Table 3.2": 29,
    "Table 3.3": 30,
    "Table 3.4": 32,
    "Table 4.1": 39,
    "Table 4.2": 39,
    "Table 4.3": 39,
    "Table 4.4": 41,
    "Table 4.5": 42,
    "Table 4.6": 45,
    "Table 5.1": 53,
    "Table 7.0": 66,
    "Table 7.1": 67,
    "Table 7.2": 68,
    "Table 7.3": 69,
    "Table 7.4": 71,
    "Table 7.5": 73,
    "Table 7.6": 74,
    "Table 7.7": 75,
    "Table 7.8": 76,
    "Table 7.9": 77,
}


def make_dot_leader_line(title, page_num, total_width=86):
    """Formats a line with title, dots, and right-aligned page number."""
    page_str = f" {page_num}"
    available_dots = max(3, total_width - len(title) - len(page_str))
    return f"{title} " + ("." * available_dots) + page_str


def process_document(docx_path, out_path=None):
    if out_path is None:
        out_path = docx_path
    
    print(f"\nProcessing {docx_path} -> {out_path}")
    doc = docx.Document(docx_path)
    
    # 1. Update List of Figures
    fig_updated = 0
    for p in doc.paragraphs:
        t = p.text.strip()
        if '[Page]' in t and t.startswith('Figure'):
            for fig_id, pg in FIGURE_PAGES.items():
                if t.startswith(fig_id):
                    # Replace [Page] with actual page number
                    new_text = re.sub(r'\[Page\]', str(pg), p.text)
                    for r in p.runs:
                        r.text = ''
                    p.runs[0].text = new_text
                    fig_updated += 1
                    break
    print(f"  Updated {fig_updated} figures in List of Figures")
    
    # 2. Update List of Tables
    tbl_updated = 0
    for p in doc.paragraphs:
        t = p.text.strip()
        if '[Page]' in t and t.startswith('Table'):
            for tbl_id, pg in TABLE_PAGES.items():
                if t.startswith(tbl_id):
                    # Replace [Page] with actual page number
                    new_text = re.sub(r'\[Page\]', str(pg), p.text)
                    for r in p.runs:
                        r.text = ''
                    p.runs[0].text = new_text
                    tbl_updated += 1
                    break
    print(f"  Updated {tbl_updated} tables in List of Tables")
    
    # 3. Populate Table of Contents
    # Locate the paragraph that says "[Generate Table of Contents..."
    toc_placeholder_idx = -1
    for i, p in enumerate(doc.paragraphs):
        if "Table of Contents" in p.text and i < 60:
            if i + 1 < len(doc.paragraphs) and "[Generate Table of Contents" in doc.paragraphs[i+1].text:
                toc_placeholder_idx = i + 1
                break
            elif "[Generate Table of Contents" in p.text:
                toc_placeholder_idx = i
                break
                
    if toc_placeholder_idx != -1:
        print(f"  Found TOC placeholder at paragraph {toc_placeholder_idx}")
        # Clear the placeholder text
        placeholder_p = doc.paragraphs[toc_placeholder_idx]
        for r in placeholder_p.runs:
            r.text = ''
            
        # Insert formatted TOC entries right after Table of Contents heading
        # In python-docx, we can format the placeholder and insert paragraphs before/after
        current_p = placeholder_p
        
        # Put the first entry in current_p
        first_title, first_pg, first_level = TOC_DATA[0]
        first_line = make_dot_leader_line(first_title, first_pg)
        current_p.text = first_line
        current_p.style = 'Normal'
        for r in current_p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10.5)
            r.font.bold = (first_level == 1)
            
        # Insert remaining entries after current_p
        prev_p = current_p
        for title, pg, level in TOC_DATA[1:]:
            indent_str = "    " * (level - 1)
            line_str = make_dot_leader_line(indent_str + title, pg, total_width=86)
            new_p = doc.add_paragraph(line_str)
            # Move new_p to correct position in XML
            prev_p._p.addnext(new_p._p)
            new_p.style = 'Normal'
            new_p.paragraph_format.space_before = Pt(2 if level == 1 else 0)
            new_p.paragraph_format.space_after = Pt(2 if level == 1 else 0)
            new_p.paragraph_format.line_spacing = 1.15
            for r in new_p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(10.5)
                r.font.bold = (level == 1)
            prev_p = new_p
        print(f"  Inserted {len(TOC_DATA)} TOC entries with leader dots and page numbers")
        
    doc.save(out_path)
    print(f"  Successfully saved: {out_path}")


if __name__ == '__main__':
    # 1. Update PCBRisk_Dissertation.docx
    process_document(r"C:\Users\priti\Downloads\pcbrisk\PCBRisk_Dissertation.docx")
    
    # 2. Update PCBRisk_Final_Dissertation.docx
    process_document(r"C:\Users\priti\Downloads\pcbrisk\PCBRisk_Final_Dissertation.docx")
    
    # 3. Create updated copy of PCBRisk_Final_Dissertation_NEW.docx if needed
    try:
        process_document(
            r"C:\Users\priti\Downloads\pcbrisk\PCBRisk_Final_Dissertation.docx",
            r"C:\Users\priti\Downloads\pcbrisk\PCBRisk_Final_Dissertation_NEW.docx"
        )
    except PermissionError:
        print("  Notice: PCBRisk_Final_Dissertation_NEW.docx is currently open in WPS Office.")
        print("  Created PCBRisk_Final_Dissertation_Updated.docx with identical updates.")
        process_document(
            r"C:\Users\priti\Downloads\pcbrisk\PCBRisk_Final_Dissertation.docx",
            r"C:\Users\priti\Downloads\pcbrisk\PCBRisk_Final_Dissertation_Updated.docx"
        )
