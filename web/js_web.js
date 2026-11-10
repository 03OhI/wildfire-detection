const video = document.getElementById('mainVideo');
const canvas = document.getElementById('resultCanvas');
const ctx = canvas.getContext('2d');
const input = document.getElementById('videoInput');
const button = document.getElementById('startBtn');
const statusText = document.getElementById('status');
const frame = document.createElement('canvas');
const frameCtx = frame.getContext('2d');
let run = 0;
let analyzing = false;
let videoUrl;

function stopAnalysis() {
    run++;
    analyzing = false;
    button.textContent = '▶ 분석 시작';
}

document.getElementById('dropZone').onclick = () => input.click();
input.onchange = () => {
    const file = input.files[0];
    if (!file) return;
    stopAnalysis();
    button.disabled = true;
    statusText.textContent = '';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    videoUrl = URL.createObjectURL(file);
    video.src = videoUrl;
};
video.onloadeddata = () => {
    frame.width = canvas.width = video.videoWidth;
    frame.height = canvas.height = video.videoHeight;
    button.disabled = false;
};
video.onerror = () => {
    stopAnalysis();
    button.disabled = true;
    statusText.textContent = '재생할 수 없는 영상입니다.';
};

for (const key of ['conf', 'iou']) {
    const slider = document.getElementById(key + 'Slider');
    slider.oninput = () => document.getElementById(key + 'Val').textContent = slider.value;
}

function drawDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 4;
    ctx.font = 'bold 20px sans-serif';
    for (const det of detections) {
        const [x, y, w, h] = det.bbox;
        ctx.fillStyle = ctx.strokeStyle = det.label === 'fire' ? '#ff534a' : '#00d7e7';
        ctx.strokeRect(x, y, w, h);
        ctx.fillText(det.label + ' ' + Math.round(det.confidence * 100) + '%', x, Math.max(y - 4, 20));
    }
}

async function analyze(id) {
    while (run === id && analyzing) {
        frameCtx.drawImage(video, 0, 0);
        const image = await new Promise(resolve => frame.toBlob(resolve, 'image/jpeg', 0.85));
        if (run !== id) return;
        if (!image) throw new Error('영상 프레임을 읽을 수 없습니다.');
        const data = new FormData();
        data.append('file', image, 'frame.jpg');
        data.append('conf_thres', document.getElementById('confSlider').value);
        data.append('iou_thres', document.getElementById('iouSlider').value);
        const response = await fetch('/analyze', { method: 'POST', body: data });
        if (!response.ok) throw new Error('분석 요청 실패 (HTTP ' + response.status + ')');
        const result = await response.json();
        if (run !== id) return;
        drawDetections(result.detections);
        statusText.textContent = '검출 ' + result.detections.length + '개';
        await new Promise(resolve => setTimeout(resolve, 100));
    }
}

button.onclick = async () => {
    if (analyzing) {
        stopAnalysis();
        return;
    }
    const id = ++run;
    analyzing = true;
    button.textContent = '⏹ 분석 중지';
    try {
        await video.play();
        await analyze(id);
    } catch (err) {
        if (run !== id) return;
        stopAnalysis();
        statusText.textContent = err.message;
    }
};
