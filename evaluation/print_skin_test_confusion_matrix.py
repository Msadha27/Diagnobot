import csv, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
eval_dir=root/'evaluation'
metrics=json.loads((eval_dir/'skin_test_metrics.json').read_text(encoding='utf-8'))['test_metrics']
with (eval_dir/'skin_test_confusion_matrix.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.reader(f))
classes=rows[0][1:]
mat=[[int(v) for v in row[1:]] for row in rows[1:]]
print('DiagnoBot | Skin classifier test-set evaluation')
print('Model: MobileNetV3-Small | Holdout: 140 images | Image-level split')
print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
print(f"Macro precision: {metrics['macro_precision']*100:.2f}%")
print(f"Balanced accuracy / macro recall: {metrics['balanced_accuracy']*100:.2f}%")
print(f"Macro F1: {metrics['macro_f1']*100:.2f}%")
print('')
label_width=max(20,max(map(len,classes)))
col_width=max(8,max(map(len,classes)))
header='True \\ Predicted'.ljust(label_width)+' '.join(c.rjust(col_width) for c in classes)
print(header)
print('-'*len(header))
for name,row in zip(classes,mat):
    print(name.ljust(label_width)+' '+' '.join(str(v).rjust(col_width) for v in row))
print('')
print('Note: internal image-level split; no patient-level or external validation.')

