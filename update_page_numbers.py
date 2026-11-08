"""
update_page_numbers.py
Uses Word COM automation to open each docx, update all field codes
(which recalculates TOC page numbers, List of Figures, List of Tables, etc.),
and save.  Falls back to a python-docx heuristic for page estimation when
Word COM is unavailable or the file is locked.
"""

import sys, os, time

def update_via_word_com(docx_path):
    """Open in Word, update all fields, save, close."""
    import win32com.client
    abs_path = os.path.abspath(docx_path)
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(abs_path)
        # Update every field in every story range (body, headers, footers)
        for story in doc.StoryRanges:
            story.Fields.Update()
        # Also update TOC if present
        for toc in doc.TablesOfContents:
            toc.Update()
        for tof in doc.TablesOfFigures:
            tof.Update()
        doc.Save()
        pages = doc.ComputeStatistics(2)  # wdStatisticPages
        doc.Close(False)
        print(f"  Word COM: updated fields & saved ({pages} pages)")
        return True
    except Exception as e:
        print(f"  Word COM error: {e}")
        try:
            doc.Close(False)
        except:
            pass
        return False
    finally:
        word.Quit()


def estimate_page_numbers_heuristic(docx_path):
    """
    python-docx doesn't know page breaks, but we can estimate page numbers
    using a rough word-count heuristic (~300 words per page for Times New Roman 12pt,
    1.15 line spacing, 1-inch margins on A4/Letter).
    
    We walk through paragraphs and tables, accumulating a running word count.
    Figures count as ~15 lines (~1/3 page).  Tables count based on row count.
    When we encounter a heading that matches a List of Figures / List of Tables
    entry, we record the estimated page number.
    """
    import docx
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl

    doc = docx.Document(docx_path)

    WORDS_PER_PAGE = 300
    FIGURE_WORD_EQUIV = 120    # a full-width figure ≈ 120 words of vertical space
    TABLE_ROW_EQUIV = 8        # each table row ≈ 8 words of vertical space

    # Collect the figure/table caption texts that appear in the body
    # and their estimated page numbers
    caption_pages = {}

    running_words = 0
    preamble_pages = 6   # title page, declaration, certificate, acknowledgement, exec summary, TOC, LoF, LoT, abbreviations

    for child in doc.element.body:
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            p = docx.text.paragraph.Paragraph(child, doc)
            text = p.text.strip()

            # Check for embedded images
            has_drawing = 'w:drawing' in child.xml
            if has_drawing:
                running_words += FIGURE_WORD_EQUIV

            if p.style.name.startswith('Heading 1'):
                # Major headings often start on new page area
                # Approximate by rounding up to next page boundary
                pass

            running_words += len(text.split())

            # Check if this is a figure or table caption in the body
            if (text.startswith('Figure 6.') or text.startswith('Figure 4.') or
                text.startswith('Figure 7.') or text.startswith('Table ')):
                # Only match actual captions (short, centered), not body text mentioning them
                if len(text) < 120 and ':' in text:
                    est_page = preamble_pages + max(1, running_words // WORDS_PER_PAGE)
                    # Extract the figure/table number (e.g. "Figure 6.1" or "Table 3.1")
                    parts = text.split(':')
                    fig_id = parts[0].strip()
                    caption_pages[fig_id] = est_page

        elif tag == 'tbl':
            tbl = docx.table.Table(child, doc)
            running_words += len(tbl.rows) * TABLE_ROW_EQUIV

    total_pages = preamble_pages + max(1, running_words // WORDS_PER_PAGE)
    print(f"  Heuristic: estimated {total_pages} total pages")
    print(f"  Found {len(caption_pages)} caption page estimates")

    # Now update the List of Figures and List of Tables entries
    for p in doc.paragraphs:
        text = p.text.strip()
        if '[Page]' not in text:
            continue

        # Find which figure/table this entry refers to
        for fig_id, page_num in caption_pages.items():
            if text.startswith(fig_id):
                old_text = p.text
                new_text = old_text.replace('[Page]', str(page_num))
                # Clear all runs and set new text
                for run in p.runs:
                    run.text = ''
                if p.runs:
                    p.runs[0].text = new_text
                else:
                    p.add_run(new_text)
                print(f"  {fig_id} -> page {page_num}")
                break

    doc.save(docx_path)
    print(f"  Saved with estimated page numbers.")


def update_page_numbers(docx_path):
    print(f"\nProcessing: {docx_path}")
    if not os.path.exists(docx_path):
        print(f"  ERROR: File not found!")
        return

    # Check if we can write to the file
    try:
        with open(docx_path, 'r+b'):
            pass
    except PermissionError:
        print(f"  File is locked (open in WPS/Word). Using heuristic on a copy...")
        return

    # Try Word COM first
    try:
        import win32com.client
        if update_via_word_com(docx_path):
            return
    except ImportError:
        print("  win32com not available, using heuristic method")

    # Fallback to heuristic
    estimate_page_numbers_heuristic(docx_path)


if __name__ == '__main__':
    # Try installing win32com
    try:
        import win32com.client
        print("win32com available!")
    except ImportError:
        print("Installing pywin32...")
        os.system(f'"{sys.executable}" -m pip install pywin32 -q')
        try:
            import win32com.client
            print("pywin32 installed successfully!")
        except ImportError:
            print("pywin32 install failed, will use heuristic method")

    files = [
        r"C:\Users\priti\Downloads\pcbrisk\PCBRisk_Dissertation.docx",
        r"C:\Users\priti\Downloads\pcbrisk\PCBRisk_Final_Dissertation_NEW.docx",
    ]
    for f in files:
        update_page_numbers(f)
