import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

# The training notebook serialized these two blocks under __main__.
# Running this file directly keeps their original names available to PyTorch.

class SSF(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c1_half = int(c1) // 2
        c2 = int(c2)

        self.cv_low = nn.Conv2d(c1_half, c2, 1)
        self.cv_high = nn.Conv2d(c1_half, c2, 1)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, c2, 1), nn.Sigmoid())

    def forward(self, x):
        f_low, f_high = torch.chunk(x, 2, dim=1)
        f_low = self.cv_low(f_low)
        f_high = self.cv_high(f_high)
        return f_low * self.ca(f_high) + f_high

# 2. EMBC 모듈 정의
class EMBC(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c1, c2 = int(c1), int(c2)
        c_ = max(1, int(c2 * e))
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 3, 1, 1, groups=c_)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_, c_, 1), nn.Sigmoid())
        self.cv3 = nn.Conv2d(c_, c2, 1, 1)

    def forward(self, x):
        feat = self.cv2(self.cv1(x))
        attn = self.se(feat)
        attended_feat = feat * attn
        return self.cv3(attended_feat)


def create_app(model):
    app = FastAPI(title="AHE-YOLO11 산불 영상 검출")

    @app.get("/")
    def read_root():
        return {"message": "AHE-YOLO11 API Server is running", "ui": "/ui/"}

    @app.post("/analyze")
    def analyze_video_frame(
        file: UploadFile = File(...),
        conf_thres: float = Form(0.25, ge=0, le=1),
        iou_thres: float = Form(0.45, ge=0, le=1),
    ):
        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(415, "이미지 파일을 보내 주세요.")
        data = np.frombuffer(file.file.read(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if frame is None:
            raise HTTPException(400, "이미지를 읽을 수 없습니다.")

        detections = []
        for result in model.predict(frame, conf=conf_thres, iou=iou_thres, verbose=False):
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "label": result.names[int(box.cls.item())],
                    "confidence": round(box.conf.item(), 2),
                    "bbox": [round(v, 2) for v in (x1, y1, x2 - x1, y2 - y1)],
                })
        return {"detections": detections}

    app.mount("/ui", StaticFiles(directory=ROOT / "web", html=True), name="ui")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="산불 화염·연기 검출 서버")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    model = YOLO(str(ROOT / "weights" / "best.pt"))
    uvicorn.run(create_app(model), host="127.0.0.1", port=args.port)
