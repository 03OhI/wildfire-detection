from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import uvicorn
import torch
import torch.nn as nn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 🚨 중요: 사용자 정의 클래스는 반드시 모델 로드 전에 위치해야 합니다.
# ==========================================

# 1. SSF 모듈 정의
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

# ==========================================
# 클래스 정의가 끝난 후, 모델을 로드합니다.
# ==========================================

try:
    model = YOLO('best.pt')
    print("✅ 모델 로드 성공")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")


@app.get("/")
def read_root():
    return {"message": "AHE-YOLO11 API Server is running"}


@app.post("/analyze")
def analyze_video_frame(
        file: UploadFile = File(...),
        conf_thres: float = Form(0.25),
        iou_thres: float = Form(0.45)
):
    try:
        if not file.content_type.startswith("image/"):
            return {"detections": [], "error": "지원하지 않는 파일 형식입니다. 이미지 파일을 업로드해주세요."}

        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"detections": [], "error": "이미지 디코딩 실패"}

        results = model.predict(
            source=frame,
            conf=conf_thres,
            iou=iou_thres
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "label": r.names[int(box.cls)],
                    "confidence": round(float(box.conf), 2),
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2)
                    ]
                })

        return {"detections": detections}

    except Exception as e:
        print(f"Server Error: {e}")
        return {"detections": [], "error": str(e)}
    finally:
        file.file.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)