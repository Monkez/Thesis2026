from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def format_docx(filename):
    doc = Document(filename)
    
    # 1. Formatting sections (margins)
    for section in doc.sections:
        section.top_margin = Cm(3.5)
        section.bottom_margin = Cm(3.0)
        section.left_margin = Cm(3.5)
        section.right_margin = Cm(2.0)
        
    # 2. Setup default style to Times New Roman, 14pt
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(14)
    
    # 3. Format all paragraphs 
    for para in doc.paragraphs:
        # Override individual empty font if it overrides style
        for run in para.runs:
            run.font.name = 'Times New Roman'
            if run.font.size is None or run.font.size < Pt(14):
                run.font.size = Pt(14)
                
        # Justify & line spacing
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para_format = para.paragraph_format
        para_format.line_spacing = 1.5
        para_format.space_after = Pt(6) # Give a small spacing after
        
    doc.save('LuanVan_De_Formatted.docx')
    print('Formatting complete! Saved as LuanVan_De_Formatted.docx')

if __name__ == '__main__':
    format_docx('LuanVan_De.docx')
