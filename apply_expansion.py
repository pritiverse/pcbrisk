# -*- coding: utf-8 -*-
"""
Script to apply the expanded academic content to PCBRisk dissertation docx files.
Preserves all existing headings, captions, tables, and figures.
"""

import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl

from expand_content_part1 import CONTENT_PART1
from expand_content_part2 import CONTENT_PART2
from expand_content_part3 import CONTENT_PART3

def build_content_dict():
    all_content = {}
    all_content.update(CONTENT_PART1)
    all_content.update(CONTENT_PART2)
    all_content.update(CONTENT_PART3)
    return all_content

def add_styled_paragraph(doc, text, style='Normal', space_after=6, line_spacing=1.15):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = line_spacing
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    return p

def expand_document(input_path, output_path):
    print(f"Loading document: {input_path}")
    doc = docx.Document(input_path)
    all_content = build_content_dict()
    
    # Map out sections in doc.element.body
    # Each section starts with a Heading CT_P
    body = doc.element.body
    
    # We will iterate through body elements and identify headings and their members
    sections = [] # list of dicts: {'heading_text': ..., 'heading_elem': ..., 'elements': []}
    current_sec = {'heading_text': '__PREAMBLE__', 'heading_elem': None, 'elements': []}
    
    for child in list(body):
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            p = docx.text.paragraph.Paragraph(child, doc)
            if p.style.name.startswith('Heading'):
                sections.append(current_sec)
                current_sec = {
                    'heading_text': p.text.strip(),
                    'heading_elem': child,
                    'elements': []
                }
                continue
        elif tag == 'sectPr':
            # section properties at the end of body
            continue
        current_sec['elements'].append(child)
    
    sections.append(current_sec)
    print(f"Identified {len(sections)} sections in document.")
    
    # Now process each section
    for sec in sections:
        h_text = sec['heading_text']
        if h_text not in all_content:
            continue
            
        new_paras_text = all_content[h_text]
        if not new_paras_text:
            continue
            
        # Special case for REFERENCES: replace existing references paragraphs
        if h_text == 'REFERENCES':
            # remove existing elements under REFERENCES
            for elem in sec['elements']:
                body.remove(elem)
            sec['elements'] = []
            
            # insert new references right after heading_elem
            anchor = sec['heading_elem']
            for ref_text in new_paras_text:
                new_p = add_styled_paragraph(doc, ref_text, space_after=4, line_spacing=1.15)
                anchor.addnext(new_p._p)
                anchor = new_p._p
            print(f"Updated REFERENCES with {len(new_paras_text)} citations.")
            continue
            
        # For all other sections: append after the last element of the section
        # If the section has elements, anchor is the last element
        # If the section has no elements, anchor is the heading element itself
        if sec['elements']:
            anchor = sec['elements'][-1]
        else:
            anchor = sec['heading_elem']
            
        # Insert each new paragraph sequentially
        for para_text in new_paras_text:
            new_p = add_styled_paragraph(doc, para_text, space_after=6, line_spacing=1.15)
            anchor.addnext(new_p._p)
            anchor = new_p._p
            sec['elements'].append(new_p._p)
            
        print(f"Expanded [{h_text}] with {len(new_paras_text)} paragraphs.")
        
    print(f"Saving expanded document to: {output_path}")
    doc.save(output_path)
    print("Save completed successfully.")

if __name__ == '__main__':
    expand_document('PCBRisk_Dissertation.docx', 'PCBRisk_Dissertation_EXPANDED_TEST.docx')
