from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from PIL import Image, ImageDraw, ImageFont
project=Path(r'C:\Users\Sadha Mail\Documents\Ding Dong bot'); sample=project/'sample_documents'; shots=sample/'project_screenshots'; ev=project/'evaluation'; report=sample/'DiagnoBot_Sample_Images_and_Figure_Captions.docx'
transcript=(ev/'skin_test_confusion_matrix_terminal.txt').read_text(encoding='utf-8-sig').splitlines(); font=ImageFont.truetype(r'C:\Windows\Fonts\consola.ttf',16); small=ImageFont.truetype(r'C:\Windows\Fonts\consola.ttf',13); pad=24; line_h=24
probe=ImageDraw.Draw(Image.new('RGB',(1,1))); width=max(int(probe.textlength(s,font=font))+pad*2 for s in transcript)+40; height=64+len(transcript)*line_h+pad
im=Image.new('RGB',(width,height),(13,17,23)); d=ImageDraw.Draw(im); d.rectangle((0,0,width,43),fill=(30,38,49))
for x,c in [(17,(248,81,73)),(38,(219,171,9)),(59,(46,160,67))]: d.ellipse((x,14,x+12,26),fill=c)
d.text((92,12),'Evaluation console output | DiagnoBot',font=small,fill=(220,225,232)); y=56
for i,line in enumerate(transcript):
 f=font if i<6 or i>=len(transcript)-1 else small; color=(201,209,217)
 if i==0: color=(121,192,255)
 if line.startswith(('Accuracy:','Macro precision:','Balanced accuracy','Macro F1:')): color=(126,231,135)
 d.text((pad,y),line,font=f,fill=color); y+=line_h
im.save(shots/'skin_test_confusion_matrix_console_output.png')
D=Document(); sec=D.sections[0]; sec.top_margin=Inches(.62); sec.bottom_margin=Inches(.62); sec.left_margin=Inches(.72); sec.right_margin=Inches(.72)
for n in ['Normal','Caption']: D.styles[n].font.name='Arial'; D.styles[n].font.size=Pt(9 if n=='Normal' else 8)
D.styles['Title'].font.name='Arial'; D.styles['Title'].font.size=Pt(23); D.styles['Title'].font.bold=True; D.styles['Title'].font.color.rgb=RGBColor(15,51,75)
D.styles['Heading 1'].font.name='Arial'; D.styles['Heading 1'].font.size=Pt(16); D.styles['Heading 1'].font.color.rgb=RGBColor(0,110,140)
def center(text,style=None):
 p=D.add_paragraph(style=style); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run(text); return p
def title(text):
 p=D.add_paragraph(); p.paragraph_format.space_after=Pt(7); r=p.add_run(text); r.bold=True; r.font.name='Arial'; r.font.size=Pt(14); r.font.color.rgb=RGBColor(15,51,75)
def add_image(path,w=6.7,h=6.15):
 with Image.open(path) as x: iw,ih=x.size
 scale=min(w/iw,h/ih); p=D.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(3); p.add_run().add_picture(str(path),width=Inches(iw*scale),height=Inches(ih*scale))
def cap(text):
 p=center(text,'Caption'); p.paragraph_format.keep_together=True
 for r in p.runs: r.italic=True; r.font.color.rgb=RGBColor(60,70,80)
def place(text):
 p=center('Suggested placement: '+text); p.paragraph_format.space_after=Pt(5)
 for r in p.runs: r.font.size=Pt(8); r.font.color.rgb=RGBColor(70,95,110)
def newpage(): D.add_page_break()
def single(heading,img,caption,where): newpage(); title(heading); add_image(img); cap(caption); place(where)
p=D.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('DiagnoBot Project Figures and Screenshots'); center('Labeled visual evidence for the prototype report')
D.add_paragraph('This figure set groups the supplied skin examples with upload-result screens when the uploaded filename is visible. Each figure includes a suggested report location. Screenshots show prototype behavior; they are not clinical validation.')
D.add_heading('Suggested report order',level=1)
for s in ['Prototype overview: Figures 1 and 2','Image upload and classifier behavior: Figures 3 to 8; keep each input beside its matching result.','Dashboard and assistant interface: Figures 9 to 12','Model training and evaluation: Figures 9 to 12','Appendix: Figure 13, the additional supplied skin-image sample.']: D.add_paragraph(s,style='List Bullet')
D.add_paragraph('The doctor dashboard screenshot with oncology claims and “10,000+ scans” is explicitly marked illustrative; these values are not supported by project evaluation files.')
single('Prototype home screen',shots/'patient_and_doctor_mode_home_screen.png','Figure 1. DiagnoBot landing screen showing Patient Mode and Doctor Mode.','System overview / Prototype design')
single('Patient AI assistant screen',shots/'patient_ai_assistant_triage_response.png','Figure 2. AI assistant interface displaying a symptom-based triage response.','Prototype interface / Assistant workflow')
# acne input and result pair
newpage(); title('Acne image and its upload result')
tab=D.add_table(rows=1,cols=2); tab.alignment=WD_TABLE_ALIGNMENT.CENTER
for cell,path,w in [(tab.cell(0,0),sample/'acne.jpg',2.8),(tab.cell(0,1),shots/'acne_upload_and_analysis_result.png',4.0)]:
 cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
 with Image.open(path) as x: iw,ih=x.size
 scale=min(w/iw,4.9/ih); cell.paragraphs[0].alignment=WD_ALIGN_PARAGRAPH.CENTER; cell.paragraphs[0].add_run().add_picture(str(path),width=Inches(iw*scale),height=Inches(ih*scale))
center('Figure 3a. Supplied acne reference image.          Figure 3b. Upload and analysis result for acne.jpg; routine result at 70% confidence.','Caption'); place('Place the supplied acne image directly beside its matching upload result in Prototype Results.')
single('Separate acne classifier result screen',shots/'upload_result_acne_routine.png','Figure 4. Acne upload-result screen showing 98% classifier confidence and routine triage. This is an application output, not an accuracy measurement.','Prototype test cases / Upload results')
single('Uncertain image result',shots/'upload_result_uncertain_low_confidence.png','Figure 5. Upload result labeled Uncertain with 26% confidence.','Prototype test cases / Low-confidence handling')
single('Fallback behavior warning',shots/'upload_result_vision_fallback_warning.png','Figure 6. Upload result showing vision-model-unavailable fallback text. It also routes the image to an ophthalmologist, which appears inconsistent with the skin-image workflow and should be corrected before presentation.','Limitations / Failure handling')
# bullous input and matching red alert result
newpage(); title('Bullous pemphigoid image and its triage result')
tab=D.add_table(rows=1,cols=2); tab.alignment=WD_TABLE_ALIGNMENT.CENTER
for cell,path,w in [(tab.cell(0,0),sample/'bullous-pemphigoid.jpg',2.8),(tab.cell(0,1),shots/'bullous_triage_alert_red_ui.png',4.0)]:
 cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
 with Image.open(path) as x: iw,ih=x.size
 scale=min(w/iw,4.9/ih); cell.paragraphs[0].alignment=WD_ALIGN_PARAGRAPH.CENTER; cell.paragraphs[0].add_run().add_picture(str(path),width=Inches(iw*scale),height=Inches(ih*scale))
center('Figure 7a. Supplied bullous pemphigoid reference image.          Figure 7b. Matching upload result for bullous-pemphigoid.jpg with red SOON alert.','Caption'); place('Keep the input and matching alert together in Prototype Results.')
single('Doctor dashboard illustrative metrics',shots/'doctor_dashboard_illustrative_metrics_not_measured.png','Figure 8. Doctor dashboard mockup. “10,000+ scans,” accuracy bars, MRI counts, and oncology alerts are not substantiated by this project’s evaluation and must not be presented as measured results.','Prototype mockup / Label as illustrative, not experimental evidence')
single('Project evaluation dashboard',shots/'project_evaluation_dashboard_measured_classifier_metrics.png','Figure 9. Dashboard displaying saved internal skin-classifier test metrics: 70.7% accuracy, 71.8% macro precision, 70.3% balanced accuracy, and 69.9% macro F1.','Evaluation results / Skin-classifier performance')
single('Training progress console',shots/'training_console_epochs.png','Figure 10. Training log showing training and validation metrics across epochs.','Methodology / Training process')
single('Model test metrics console',shots/'test_metrics_console_output.png','Figure 11. Console output with held-out test-set metrics and checkpoint details.','Evaluation results / Test-set metrics')
# matrix in landscape
land=D.add_section(start_type=2); land.orientation=WD_ORIENT.LANDSCAPE; land.page_width=Inches(11); land.page_height=Inches(8.5); land.top_margin=Inches(.45); land.bottom_margin=Inches(.45); land.left_margin=Inches(.55); land.right_margin=Inches(.55)
title('Held-out test confusion matrix'); add_image(shots/'skin_test_confusion_matrix_console_output.png',9.8,6.5); cap('Figure 12. Console transcript generated by print_skin_test_confusion_matrix.py from the saved test metrics and confusion-matrix CSV. Rows are true labels; columns are predicted labels.'); place('Evaluation results / Confusion matrix. The test set has 140 images and uses an image-level split.')
port=D.add_section(start_type=2); port.orientation=WD_ORIENT.PORTRAIT; port.page_width=Inches(8.5); port.page_height=Inches(11); port.top_margin=Inches(.62); port.bottom_margin=Inches(.62); port.left_margin=Inches(.72); port.right_margin=Inches(.72)
single('Additional supplied skin test image',sample/'acne1.jpg','Figure 13. Additional supplied skin-image sample. No result is paired because the supplied screenshots do not show this filename with a corresponding analysis.','Appendix / Sample test images')
newpage(); D.add_heading('Source files and use notes',level=1); D.add_paragraph('The sample_documents folder contains the supplied one-page blood-test PDF, skin images, this figure document, and the unique copied project screenshots. Use the suggested placement under each figure when adding it to the project report.')
D.add_paragraph('The classifier metrics describe the saved 140-image internal test split only. They do not measure clinical safety, triage quality, assistant-answer quality, or performance on new patients. Some screenshots show earlier prototype behavior.')
D.save(report); print(report)






