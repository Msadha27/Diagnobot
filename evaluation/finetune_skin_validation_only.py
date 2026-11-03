from __future__ import annotations
import argparse, csv, json, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

p=argparse.ArgumentParser(); p.add_argument('--project',required=True); args=p.parse_args()
root=Path(args.project); data=root/'data'/'skin_diseases'; models_dir=root/'models'; eval_dir=root/'evaluation'
source=models_dir/'skin_mobilenet.pt'; cand=models_dir/'skin_mobilenet_finetuned_val.pt'
seed=42; size=224; batch=32; max_epochs=12; patience=4
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(max(1,min(8,torch.get_num_threads())))
device=torch.device('cpu')
train_tf=transforms.Compose([transforms.Resize((size,size)),transforms.RandomHorizontalFlip(.5),transforms.RandomRotation(8),transforms.ColorJitter(brightness=.10,contrast=.10,saturation=.08),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
eval_tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
base=datasets.ImageFolder(str(data)); classes=base.classes
by=defaultdict(list)
for i,y in enumerate(base.targets): by[y].append(i)
rng=random.Random(seed); tr=[]; va=[]
for y in sorted(by):
    ids=by[y][:]; rng.shuffle(ids); n=len(ids); nt=round(.70*n); nv=round(.15*n); tr.extend(ids[:nt]); va.extend(ids[nt:nt+nv])
rng.shuffle(tr); rng.shuffle(va)
trainbase=datasets.ImageFolder(str(data),transform=train_tf); evalbase=datasets.ImageFolder(str(data),transform=eval_tf)
train=Subset(trainbase,tr); val=Subset(evalbase,va)
trainloader=DataLoader(train,batch_size=batch,shuffle=True,num_workers=0); valloader=DataLoader(val,batch_size=batch,shuffle=False,num_workers=0)
def make_model():
    m=models.mobilenet_v3_small(weights=None); m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,len(classes)); return m

def score(m,loader):
    m.eval(); yt=[]; yp=[]; loss_sum=0.; crit=nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x,y in loader:
            z=m(x); loss_sum+=crit(z,y).item()*y.size(0); yt.extend(y.tolist()); yp.extend(z.argmax(1).tolist())
    n=len(classes); cm=[[0]*n for _ in range(n)]
    for a,b in zip(yt,yp): cm[a][b]+=1
    pc=[]
    for i in range(n):
        tp=cm[i][i]; sup=sum(cm[i]); fp=sum(cm[j][i] for j in range(n))-tp; fn=sup-tp
        pr=tp/(tp+fp) if tp+fp else 0.; re=tp/sup if sup else 0.; f=2*pr*re/(pr+re) if pr+re else 0.
        pc.append({'precision':pr,'recall':re,'f1':f,'support':sup})
    return {'accuracy':sum(cm[i][i] for i in range(n))/max(1,len(yt)),'macro_precision':sum(a['precision'] for a in pc)/n,'macro_recall':sum(a['recall'] for a in pc)/n,'balanced_accuracy':sum(a['recall'] for a in pc)/n,'macro_f1':sum(a['f1'] for a in pc)/n,'loss':loss_sum/max(1,len(yt)),'per_class':dict(zip(classes,pc)),'confusion_matrix':cm,'n':len(yt)}
# Validation-only baseline. No test images are decoded or scored by this script.
ck=torch.load(source,map_location=device,weights_only=False)
if ck.get('class_names')!=classes: raise ValueError('Checkpoint class order differs from ImageFolder class order')
model=make_model(); model.load_state_dict(ck['model_state_dict']); model.to(device)
baseline=score(model,valloader)
print(f"VALIDATION BASELINE: acc={baseline['accuracy']:.4f} macroF1={baseline['macro_f1']:.4f} n={baseline['n']}",flush=True)
# Continue from the current active model with conservative fine-tuning and frozen early BN statistics.
blocks=list(model.features.children())
for q in model.features.parameters(): q.requires_grad=False
for block in blocks[-4:]:
    for q in block.parameters(): q.requires_grad=True
for q in model.classifier.parameters(): q.requires_grad=True
counts=np.bincount([base.targets[i] for i in tr],minlength=len(classes))
weights=torch.tensor(np.sqrt(counts.sum()/counts),dtype=torch.float32)
criterion=nn.CrossEntropyLoss(weight=weights,label_smoothing=.03)
optimizer=torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],lr=8e-6,weight_decay=3e-4)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2)
best=None; best_key=(-1.,-1.); best_epoch=0; wait=0; hist=[]; start=time.time()
for epoch in range(1,max_epochs+1):
    model.train()
    for b in blocks[:-4]: b.eval()  # freeze running statistics for frozen backbone
    loss_sum=0.; correct=0; seen=0
    for x,y in trainloader:
        optimizer.zero_grad(set_to_none=True); logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step()
        loss_sum+=loss.item()*y.size(0); correct+=(logits.argmax(1)==y).sum().item(); seen+=y.size(0)
    vm=score(model,valloader); scheduler.step(vm['macro_f1'])
    row={'epoch':epoch,'train_loss':loss_sum/max(1,seen),'train_accuracy':correct/max(1,seen),'val_accuracy':vm['accuracy'],'val_macro_precision':vm['macro_precision'],'val_balanced_accuracy':vm['balanced_accuracy'],'val_macro_f1':vm['macro_f1'],'val_loss':vm['loss'],'lr':optimizer.param_groups[0]['lr']}; hist.append(row)
    print(f"Epoch {epoch:02d}: train_acc={row['train_accuracy']:.4f} val_acc={row['val_accuracy']:.4f} val_macroF1={row['val_macro_f1']:.4f} val_loss={row['val_loss']:.4f}",flush=True)
    key=(vm['macro_f1'],vm['accuracy'])
    if key>best_key:
        best_key=key; best_epoch=epoch; wait=0
        best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save({'model_state_dict':best,'class_names':classes,'img_size':size,'architecture':'mobilenet_v3_small','best_epoch':epoch,'best_val_macro_f1':vm['macro_f1'],'best_val_accuracy':vm['accuracy'],'selection':'validation macro-F1, accuracy tie-break; test not used'},cand)
        best_metrics=vm
    else:
        wait+=1
    if wait>=patience:
        print('Early stopping: validation score did not improve.',flush=True); break
# Persist validation-only logs and both validation confusion matrices.
with (eval_dir/'skin_finetune_validation_history.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
for tag,sc in [('baseline',baseline),('candidate',best_metrics)]:
    with (eval_dir/f'skin_finetune_{tag}_validation_confusion_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['true \\ predicted',*classes]);
        for name,row in zip(classes,sc['confusion_matrix']): w.writerow([name,*row])
summary={'dataset_images':len(base),'classes':classes,'train_images':len(tr),'validation_images':len(va),'test_partition_used':False,'seed':seed,'selection':'highest validation macro-F1; accuracy tie-break','baseline_validation':baseline,'candidate_validation':best_metrics,'candidate_checkpoint':str(cand),'best_epoch':best_epoch,'elapsed_seconds':round(time.time()-start,1),'notes':['The test partition was not loaded or evaluated in this tuning run.','The 140-image test split has already been reported previously and remains unsuitable for tuning.','This is a single validation-guided fine-tuning run; confirm on a new patient/source-independent cohort.']}
(eval_dir/'skin_finetune_validation_comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('SELECTED VALIDATION CHECKPOINT',cand,flush=True)
print('BASELINE VAL',json.dumps({k:v for k,v in baseline.items() if k not in ('per_class','confusion_matrix')},indent=2),flush=True)
print('CANDIDATE VAL',json.dumps({k:v for k,v in best_metrics.items() if k not in ('per_class','confusion_matrix')},indent=2),flush=True)
