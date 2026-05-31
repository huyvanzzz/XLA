# Engineering Guide

## Default Validation Commands

Run after edits:

```powershell
python -m py_compile train.py predict.py utils\config.py utils\dataset.py utils\loss.py utils\inference.py utils\box_ops.py models\tiny_detector.py
python -m unittest test_predict_cli.py
```

If touching model/head/neck:

```powershell
@'
import torch
from utils.config import load_config
from models.tiny_detector import TinyDetector
c=load_config('configs/default.yaml')
mc=dict(c['model']); mc['pretrained']=False
m=TinyDetector(num_classes=5,num_anchors=[3,3,3],**mc)
x=torch.randn(1,3,512,512)
y=m(x)
print(type(m.neck).__name__, type(m.main_heads[0]).__name__)
print([tuple(t.shape) for t in y['main']])
print('params', sum(p.numel() for p in m.parameters()))
'@ | python -
```

If touching loss:

```powershell
@'
import torch
from utils.loss import YoloLoss
anchors=[[(20.,30.),(40.,50.),(80.,90.)],[(100.,120.),(140.,160.),(180.,200.)],[(220.,240.),(280.,300.),(340.,360.)]]
criterion=YoloLoss(anchors,512,5,decode_style='yolov7',target_offsets=True,scale_obj_balance=[4.0,1.0,0.4],objectness_iou_mix=1.0,box_weight=0.0,iou_weight=6.0)
preds=[torch.randn(2,64,64,3,10,requires_grad=True),torch.randn(2,32,32,3,10,requires_grad=True),torch.randn(2,16,16,3,10,requires_grad=True)]
targets=[{'boxes':torch.tensor([[50.,60.,160.,200.],[250.,120.,360.,300.]]),'labels':torch.tensor([0,4])},{'boxes':torch.empty((0,4)),'labels':torch.empty((0,),dtype=torch.long)}]
loss,logs=criterion({'main':preds,'aux':[]},targets)
loss.backward()
print(float(loss.detach()), logs)
'@ | python -
```

## Safe Directions

Prefer changes with measurable effect and limited train-time cost:

- post-processing and per-class threshold after training;
- NMS candidate limits;
- ResNet50 + local neck/head changes;
- loss/decode changes that do not require global candidate assignment;
- checkpoint/resume robustness for Kaggle;
- small ablations one at a time.

## Risky Directions

Avoid or discuss before using:

- task-aligned assignment over all candidates: slow;
- OTA matching: slow and tied to YOLOv7 Detect head;
- ConvNeXt-Small/Base with fine-tuning: heavier, may OOM or slow train;
- LayerNorm/permute-heavy neck/head: can be slower despite fewer params;
- mosaic/hard-negative/TTA defaults: previously bad;
- letterbox: user rejected it;
- validation-time threshold grid every epoch: too slow.

## YOLOv7 Adaptation Rules

It is okay to read `WongKinYiu/yolov7` for architectural ideas, but do not copy the complete detector.

Allowed local adaptations already used:

- SPPCSPC-like P5 context block;
- ELAN-style multi-branch fusion;
- maxpool + stride-2 conv downsample route;
- bbox decode formula `sigmoid*2-0.5` and `(sigmoid*2)^2*anchor`;
- target offsets to adjacent cells;
- objectness scale balance `[4.0, 1.0, 0.4]`.

Avoid:

- importing YOLOv7 modules directly;
- using YOLOv7 model parser;
- using YOLOv7 Detect/IDetect;
- replacing this repo train/predict/loss wholesale.

## mAP Interpretation Notes

- Low micro precision with high recall can still have decent AP if high-confidence predictions rank well.
- mAP@0.5 is area under per-class precision-recall curve, averaged over classes.
- For this dataset, `chair` has historically been weak.
- If mAP plateaus near `0.61`, inspect per-class AP and prediction counts before changing architecture.

## Kaggle Notes

- Save checkpoints under `/kaggle/working` or configured `checkpoint_dir`.
- Current training does not save `best.pth` before mAP starts at epoch 30.
- If crash risk matters, add periodic `last.pth` saves without using them as best.
- `Save & Run All` reruns the notebook; quick save/version behavior differs.
