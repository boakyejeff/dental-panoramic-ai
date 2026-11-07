"""Benchmark detector configs on CPU to pick a feasible 705-image run.

Times one train step (forward+backward+step) and one inference pass on
real DENTEX val images, for a few backbone/resolution/freeze combos.
Usage: ~/workspace/venvs/crop-disease/bin/python src/bench_detector.py
"""
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
torch.set_num_threads(4)


def to_tensor(path, size):
    im = Image.open(path).convert("RGB").resize((size[1], size[0]))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def make_target():
    # one dummy caries box, roughly centred, in resized coords
    return {"boxes": torch.tensor([[200.0, 100.0, 300.0, 200.0]]),
            "labels": torch.tensor([2], dtype=torch.int64)}


def bench(name, make_model, size, freeze_backbone, n_steps=3):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    img = sorted((DATA / "cache1024" / "dentex").glob("*.jpg"))[0]
    model = make_model()
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 5)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005, momentum=0.9)
    model.train()
    x = to_tensor(img, size)
    tgt = make_target()
    # rescale dummy box to this size
    sx, sy = size[1] / 1024, size[0] / 512
    tgt["boxes"] = tgt["boxes"] * torch.tensor([sx, sy, sx, sy])
    t0 = time.time()
    for _ in range(n_steps):
        loss_dict = model([x], [tgt])
        loss = sum(loss_dict.values())
        opt.zero_grad(); loss.backward(); opt.step()
    train_s = (time.time() - t0) / n_steps
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        for _ in range(n_steps):
            _ = model([x])
        inf_s = (time.time() - t0) / n_steps
    print(f"{name} size={size} frozen={freeze_backbone}: "
          f"train {train_s:.1f}s/img  infer {inf_s:.1f}s/img  "
          f"-> 705 img/epoch ~{train_s * 705 / 60:.0f} min", flush=True)
    del model
    import gc; gc.collect()


def resnet50():
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    return fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)


def mobilenet():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
    return fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)


if __name__ == "__main__":
    bench("resnet50-fpn 1024x512 (pilot config)", resnet50, (512, 1024), False)
    bench("resnet50-fpn 768x384 frozen", resnet50, (384, 768), True)
    bench("mobilenetv3-fpn 768x384 frozen", mobilenet, (384, 768), True)
    bench("mobilenetv3-fpn 512x256 frozen", mobilenet, (256, 512), True)
