#!/usr/bin/env python3
"""Render notebook-produced figures and draft text as a self-contained meeting PDF."""
import argparse
from pathlib import Path
import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, KeepTogether,
    Table, TableStyle,
)


def build(root, manifest_path, output):
    root=Path(root);data=json.loads(Path(manifest_path).read_text())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
    styles=getSampleStyleSheet()
    for name in ['Normal','BodyText','Heading1','Heading2','Title']:
        styles[name].fontName='Body' if name in ['Normal','BodyText'] else 'Bold'
    styles['BodyText'].fontSize=9.5;styles['BodyText'].leading=14
    styles['Title'].fontSize=26;styles['Title'].leading=32;styles['Title'].alignment=TA_LEFT
    styles['Title'].textColor=colors.HexColor('#173e45')
    styles['Heading1'].fontSize=18;styles['Heading1'].leading=23
    styles['Heading2'].fontSize=12;styles['Heading2'].leading=17
    small=ParagraphStyle('Caption',parent=styles['BodyText'],fontSize=8.5,leading=12,
                         textColor=colors.HexColor('#3e4c59'))
    eyebrow=ParagraphStyle('Eyebrow',parent=small,fontName='Bold',fontSize=9,
                           textColor=colors.HexColor('#287d78'),spaceAfter=14)
    W,H=landscape(A4);margin=36;avail_w=W-2*margin-12
    story=[]
    def paragraph(s,style='BodyText'):
        return Paragraph(escape(s).replace('\n',' '),styles[style])
    def header(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#cbd5e1'));canvas.setLineWidth(.6)
        canvas.line(margin,H-26,W-margin,H-26)
        canvas.setFont('Body',8);canvas.setFillColor(colors.HexColor('#52616f'))
        canvas.drawString(margin,H-18,'PCB component detection | Experimental evidence | 6 November 2026')
        canvas.drawRightString(W-margin,17,f'{doc.page}')
        canvas.drawString(margin,17,'Meeting draft - completed evaluations; pending experiments are excluded.')
    story.extend([
        Paragraph('PROFESSOR MEETING / RESEARCH DRAFT',eyebrow),
        paragraph('Experimental Setup and Results','Title'),
        paragraph('Reproducible figures, verified numbers, and a manuscript starting point.','Heading2'),
        Spacer(1,12),
    ])
    primary=data['primary']
    def row(m):return next(r for r in primary if r['model']==m)
    cards=[
        ['Completed original-label study','Crop-trained AGWD + Dice','Observed change vs YOLO26s'],
        ['6 methods x 3 seeds',data['crop_mAP50']+'% mAP50',f"+{data['gain_vs_yolo26_pp']:.2f} percentage points"],
        ['543 train / 77 validation / 41 test exports','Mean +/- sample SD; not a confidence interval','Shared evaluator; no significance claim'],
    ]
    table=Table(cards,colWidths=[avail_w*.33,avail_w*.35,avail_w*.32])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#edf5f3')),
        ('TEXTCOLOR',(0,0),(-1,-1),colors.HexColor('#173e45')),
        ('FONTNAME',(0,0),(-1,-1),'Body'),('FONTNAME',(0,1),(-1,1),'Bold'),
        ('FONTSIZE',(0,0),(-1,0),9),('FONTSIZE',(0,1),(-1,1),15),
        ('FONTSIZE',(0,2),(-1,2),7.7),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),
        ('LEFTPADDING',(0,0),(-1,-1),11),
    ]))
    story.extend([table,Spacer(1,20)])
    paragraphs=[
        ('What can be presented today',
         'Completed original-label three-seed comparisons, class metrics, saved-prediction PR curves, confusion matrices, '
         'qualitative detections, actual training histories, isolated runtime, and traced annotation edits.'),
        ('How to frame the older native-resolution results',
         'The retrospective champion/ensemble audit is exploratory: cross-split photographs and annotation defects were '
         'confirmed. The original-label study also inherits three shared view-name aliases across splits; its matched '
         'comparisons are benchmark-specific, not certified board-disjoint generalization.'),
        ('What is still pending',
         'Corrected-r3 training and the new AGWD/Exp7 runs have no completed results in this handout. Exact equivalence '
         'to the published paper evaluator and image annotations is unverified. Published scores are contextual only.'),
        ('Suggested writing order',
         'Draft datasets, implementation, evaluation metrics and the evidence-backed results first. The manuscript normally '
         'presents Introduction, Related Work, Method, Experimental Setup and Results, Discussion, and Conclusion.'),
    ]
    for title,body in paragraphs:
        story.extend([paragraph(title,'Heading2'),paragraph(body),Spacer(1,8)])
    story.append(PageBreak())
    for i,f in enumerate(data['figures'],1):
        title=paragraph(f"Figure {i}. {f['title']}",'Heading1')
        scope=Paragraph(escape(f['scope']),eyebrow)
        caption=Paragraph(escape(f['caption']),small)
        im=Image(str(root/f['png']))
        used=sum(p.wrap(avail_w,10000)[1]+p.getSpaceBefore()+p.getSpaceAfter()
                 for p in [title,scope,caption])+24
        max_w=avail_w;max_h=H-42-30-12-used
        assert max_h>200, f['id']
        scale=min(max_w/im.imageWidth,max_h/im.imageHeight)
        im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale
        im.hAlign='CENTER'
        story.extend([KeepTogether([title,scope,im,Spacer(1,10),caption]),PageBreak()])
    story.append(paragraph('Experimental section - prose draft','Heading1'))
    text=(root/data['draft']).read_text()
    for part in text.split('\n\n'):
        if part.startswith('# '):continue
        if part.startswith('## '):
            heading,_,body=part.partition('\n')
            story.append(Spacer(1,9));story.append(paragraph(heading[3:],'Heading2'))
            if body:story.extend([paragraph(body),Spacer(1,8)])
        else:
            story.extend([paragraph(part),Spacer(1,8)])
    story.extend([PageBreak(),paragraph('Reproducibility and scope','Heading1')])
    for text in [
        'Open paper/notebooks/experiments_and_results.ipynb to rerun the figures without GPU inference. '
        'Saved predictions and original labels are checked against recorded hashes.',
        'All 18 common-profile seed summaries are checked against the saved aggregate mean/sample-SD tables. '
        'The 24 seed-0 class AP50 results and fixed-confidence TP/FP counts are independently recomputed.',
        'Tables preserve full numeric precision in CSV. Figures also export editable SVG. '
        'The figure manifest records the source file hashes, software versions, scope and captions.',
        'Background columns in diagnostic confusion matrices are independently normalized FP compositions. '
        'They must not be interpreted as background recall or substituted for class-aware AP matching.',
        'The three-seed mean improvements are descriptive. No p-values, invented confidence intervals, '
        'best-seed selection or unsupported improvement against published-paper numbers is included.',
    ]:story.extend([paragraph(text),Spacer(1,10)])
    story.append(paragraph('Source reference','Heading2'))
    story.append(paragraph("X. Zhou and S. Agaian. Ensemble Learning Using YOLO Models for Semiconductor "
        "E-Waste Recycling. Information 2026, 17, 322. https://doi.org/10.3390/info17040322"))
    doc=SimpleDocTemplate(str(output),pagesize=(W,H),rightMargin=margin,leftMargin=margin,
        topMargin=42,bottomMargin=30,title='PCB Detection - Experimental Setup and Results',
        author='PCB detection research project')
    doc.build(story,onFirstPage=header,onLaterPages=header)
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.root,a.manifest,a.output)
