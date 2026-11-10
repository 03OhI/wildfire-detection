const video = document.getElementById('mainVideo');
const canvas = document.getElementById('resultCanvas');
const ctx = canvas.getContext('2d');
const videoInput = document.getElementById('videoInput');
const dropZone = document.getElementById('dropZone');
const startBtn = document.getElementById('startBtn');

let isAnalyzing = false;

// 1. 영상 로딩
dropZone.onclick = () => videoInput.click();
videoInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
        video.src = URL.createObjectURL(file);
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            console.log("✅ 비디오 로드 완료:", canvas.width, "x", canvas.height);
        };
        video.play();
    }
};

// 2. 슬라이더 업데이트
document.getElementById('confSlider').oninput = (e) => document.getElementById('confVal').innerText = e.target.value;
document.getElementById('iouSlider').oninput = (e) => document.getElementById('iouVal').innerText = e.target.value;

// 3. 실시간 분석 루프
async function analyzeLoop() {
    if (!isAnalyzing || video.paused || video.ended) {
        console.log("⏹ 분석 중지됨");
        return;
    }

    // 현재 프레임을 캡처할 오프스크린 캔버스
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);

    // 4. FastAPI 전송
    tempCanvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        formData.append('conf_thres', document.getElementById('confSlider').value);
        formData.append('iou_thres', document.getElementById('iouSlider').value);

        try {
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            // 데이터가 왔는지 확인하는 로그
            console.log("📡 서버 응답:", data.detections.length, "개의 객체 탐지됨");
            
            drawDetections(data.detections);
        } catch (err) {
            console.error("❌ 서버 통신 에러:", err);
        }

        // 반복 실행 (100ms 간격)
        if (isAnalyzing) setTimeout(analyzeLoop, 100);
    }, 'image/jpeg', 0.8);
}

// 5. 화면에 그리기
function drawDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let fireConf = [];
    let smokeConf = [];
    detections.forEach(det => {
        const [x, y, w, h] = det.bbox;
        ctx.strokeStyle = '#ff4d4d'; // 산불이니까 빨간색
        ctx.lineWidth = 5;
        ctx.strokeRect(x, y, w, h);

        // 텍스트 배경
        ctx.fillStyle = '#ff4d4d';
        ctx.font = 'bold 24px Arial';
        ctx.fillText(`${det.label} ${Math.round(det.confidence * 100)}%`, x, y - 10);
        if (det.label === 'fire') fireConf.push(det.confidence);
        else if (det.label === 'smoke') smokeConf.push(det.confidence);
    });
    const avgFire = fireConf.length ? fireConf.reduce((a,b) => a+b)/fireConf.length : 0;
    const avgSmoke = smokeConf.length ? smokeConf.reduce((a,b) => a+b)/smokeConf.length : 0;
    
    confChart.data.datasets[0].data = [avgFire, avgSmoke];
    confChart.update('none'); // 애니메이션 없이 빠르게 업데이트
}

// 6. 시작 버튼 클릭
startBtn.onclick = () => {
    isAnalyzing = !isAnalyzing;
    startBtn.innerText = isAnalyzing ? "⏹ 분석 중지" : "▶ 분석 시작";
    startBtn.style.background = isAnalyzing ? "#ef4444" : "#3b82f6";
    
    if (isAnalyzing) {
        console.log("🚀 분석 루프 시작");
        analyzeLoop();
    }
};