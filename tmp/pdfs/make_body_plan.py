from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.colors import HexColor, white
from pathlib import Path
out=Path(r'C:\Users\Sadha Mail\Documents\Ding Dong bot\output\pdf\DiagnoBot_Body_Measurement_Plan_1page.pdf'); W,H=landscape(A4)
c=canvas.Canvas(str(out),pagesize=(W,H)); R=HexColor('#c62828'); K=HexColor('#171717'); G=HexColor('#555555'); P=HexColor('#f7f7f7')
def line(a,b,d,e,col=K,w=.8,dash=None):
 c.setStrokeColor(col); c.setLineWidth(w); c.setDash(dash or []); c.line(a,b,d,e); c.setDash()
def t(x,y,s,z=7,col=K,f='Helvetica'):
 c.setFillColor(col); c.setFont(f,z); c.drawString(x,y,s)
def ct(x,y,s,z=7,col=K,f='Helvetica'):
 c.setFillColor(col); c.setFont(f,z); c.drawCentredString(x,y,s)
def panel(x,y,w,h,title,sub):
 c.setFillColor(white); c.setStrokeColor(K); c.roundRect(x,y,w,h,7,stroke=1,fill=1); t(x+10,y+h-17,title,10,K,'Helvetica-Bold'); t(x+10,y+h-30,sub,7,G)
def dh(x1,x2,y,ey,label):
 line(x1,ey,x1,y-2,R,.6); line(x2,ey,x2,y-2,R,.6); line(x1,y,x2,y,R,1)
 for x,d in ((x1,1),(x2,-1)): line(x,y,x+d*4,y+2,R,1); line(x,y,x+d*4,y-2,R,1)
 ct((x1+x2)/2,y+3,label,8,R,'Helvetica-Oblique')
def dv(x,y1,y2,ex,label):
 line(ex,y1,x+2,y1,R,.6); line(ex,y2,x+2,y2,R,.6); line(x,y1,x,y2,R,1)
 for y,d in ((y1,1),(y2,-1)): line(x,y,x-2,y+d*4,R,1); line(x,y,x+2,y+d*4,R,1)
 c.saveState(); c.translate(x-4,(y1+y2)/2); c.rotate(90); ct(0,0,label,8,R,'Helvetica-Oblique'); c.restoreState()
c.setFillColor(white); c.rect(0,0,W,H,fill=1,stroke=0)
t(28,H-32,'DIAGNOBOT | BODY BUILD MEASUREMENTS',17,K,'Helvetica-Bold'); t(29,H-47,'Proposed enclosure: 300 x 300 x 250 mm overall | 5 mm plywood | dimensions in mm',8,G); line(28,H-55,W-28,H-55,K,1)
# front
panel(28,323,382,206,'FRONT VIEW','300 mm wide x 250 mm body height; chassis and wheels add height below.')
x,y,w,h=57,355,150,120;c.setFillColor(P);c.setStrokeColor(K);c.setLineWidth(1.2);c.rect(x,y,w,h,fill=1)
c.setStrokeColor(G);c.setDash(3,2);c.rect(x+45,y+38,60,38,fill=0);c.setDash();ct(x+75,y+58,'LCD RESERVE',7,K,'Helvetica-Bold');ct(x+75,y+47,'120 x 75 zone only',6.5,G);c.setStrokeColor(K);c.circle(x+75,y+100,3,fill=0,stroke=1);ct(x+75,y+88,'camera guide',6,G)
dh(x,x+w,y-11,y,'300');dv(x-12,y,y+h,x,'250')
t(226,449,'Front panel: 300 x 240 x 5',8,K,'Helvetica-Bold');t(226,432,'LCD and camera openings are',7,G);t(226,421,'placeholders. Measure actual parts',7,G);t(226,410,'before cutting.',7,K,'Helvetica-Oblique')
# top
panel(432,323,382,206,'TOP VIEW | LID OFF','Outer footprint 300 x 300 mm; front / display edge shown at bottom.')
x,y,s=461,347,138;c.setFillColor(P);c.setStrokeColor(K);c.setLineWidth(1.2);c.rect(x,y,s,s,fill=1);c.setStrokeColor(G);c.setDash(3,2);c.rect(x+7,y+7,124,124,fill=0);c.setDash()
for bx,by,bw,bh,a,b in [(x+15,y+66,48,35,'Pi 5','110 x 80'),(x+74,y+89,40,25,'ESP32','70 x 50'),(x+70,y+19,47,33,'L298N','100 x 70')]:
 c.setFillColor(white);c.setStrokeColor(K);c.rect(bx,by,bw,bh,fill=1);ct(bx+bw/2,by+bh/2+2,a,6.5,K,'Helvetica-Bold');ct(bx+bw/2,by+bh/2-6,b,5.8,G)
ct(x+s/2,y+3,'FRONT / DISPLAY',6,K,'Helvetica-Bold');dh(x,x+s,y-11,y,'300');dv(x-12,y,y+s,x,'300')
t(615,463,'Upper tray: 270 x 240 mm (3 mm).',7.3,K,'Helvetica-Bold');t(615,449,'Dashed outline = tray boundary.',7,G);t(615,435,'Boxes show approximate keep-clear',7,G);t(615,424,'zones, not mounting-hole patterns.',7,G);t(615,398,'Keep Pi ports and airflow clear.',7,G);t(615,386,'Route motor wires away from sensors.',7,G)
# side
panel(28,105,382,206,'SIDE VIEW','300 mm deep x 250 mm body height; front is left, rear is right.')
x,y,w,h=60,139,145,120;c.setFillColor(P);c.setStrokeColor(K);c.setLineWidth(1.2);c.rect(x,y,w,h,fill=1);line(x+3,y+h*.18,x+w-3,y+h*.18,K,1.5);line(x+3,y+h*.52,x+w-3,y+h*.52,K,1.5);c.setStrokeColor(G);c.setDash(3,2);c.rect(x+5,y+h*.42,50,28,fill=0);c.setDash();dh(x,x+w,y-11,y,'300');dv(x-12,y,y+h,x,'250')
t(226,223,'Upper tray level: 135 mm',8,R,'Helvetica-Oblique');t(226,210,'Lower tray level: 45 mm',8,R,'Helvetica-Oblique');t(226,184,'Measure levels from inside floor.',7,G);t(226,171,'Adjust after checking battery size,',7,G);t(226,158,'connectors, airflow and cable bends.',7,G)
# internal
panel(432,105,382,206,'INTERNAL CUTAWAY','Front wall removed; shelves removable; component blocks are placement zones.')
x,y,w,h=461,130,154,126;c.setStrokeColor(K);c.setLineWidth(3);c.rect(x,y,w,h,fill=0);line(x+3,y+h*.32,x+w-3,y+h*.32,K,2);line(x+3,y+h*.67,x+w-3,y+h*.67,K,2)
for bx,bw,a in [(x+10,49,'Pi 5'),(x+64,38,'ESP32'),(x+107,37,'L298N')]:
 c.setFillColor(white);c.setStrokeColor(K);c.setLineWidth(.7);c.rect(bx,y+h*.69,bw,h*.19,fill=1);ct(bx+bw/2,y+h*.77,a,6.2,K,'Helvetica-Bold')
c.setStrokeColor(G);c.setDash(3,2);c.rect(x+45,y+10,65,23,fill=0);c.setDash();ct(x+77.5,y+19,'BATTERY ZONE',6.5,K,'Helvetica-Bold')
t(634,235,'Lower tray: 270 x 270 mm (3 mm)',7.3,K,'Helvetica-Bold');t(634,220,'Battery kept low and centered.',7,G);t(634,188,'Upper tray: 270 x 240 mm (3 mm)',7.3,K,'Helvetica-Bold');t(634,173,'Pi zone 110 x 80 | ESP32 70 x 50',7,G);t(634,161,'L298N zone 100 x 70 mm.',7,G);t(634,135,'Leave vents and connectors accessible.',7,K,'Helvetica-Oblique')
# cutlist
c.setFillColor(P);c.setStrokeColor(K);c.roundRect(28,24,W-56,68,6,fill=1);t(38,77,'CUT LIST | 5 mm plywood body; 3 mm removable trays',9,K,'Helvetica-Bold')
rows=[('Base + removable lid','300 x 300','2'),('Front + back walls','300 x 240','2'),('Left + right walls','290 x 240','2'),('Lower tray (3 mm)','270 x 270','1'),('Upper tray (3 mm)','270 x 240','1')]
for i,(a,b,d) in enumerate(rows):
 yy=63-i*9;t(39,yy,a,6.6);t(204,yy,b,6.8,R,'Helvetica-Oblique');t(295,yy,d,6.6)
t(355,62,'BEFORE CUTTING',7.4,K,'Helvetica-Bold');t(355,50,'Confirm chassis, wheel clearance, battery, display, webcam,',6.5);t(355,40,'and cable bends. Adapt base mounting holes to your chassis.',6.5);t(355,30,'Proposed enclosure plan, not a verified chassis drawing.',6.5,G,'Helvetica-Oblique');ct(W/2,12,'SCHEMATIC | NOT TO SCALE | Check hardware physically before cutting.',6.5,G,'Helvetica-Oblique')
c.showPage();c.save();print(out)

