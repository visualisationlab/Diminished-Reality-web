const MOTION_LONGEST = 96;
const SEARCH_RADIUS = 6;
const SAMPLE_STEP = 2;
const BORDER = 6;
const MIN_TEXTURE = 10;
const MIN_IMPROVEMENT = 1.25;
const MAX_COMPENSATION_FRACTION = 0.25;
const MOTION_MIN_UPDATE_MS = 33;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function scoreShift(prevGray, currGray, width, height, dx, dy) {
  const xStart = BORDER + Math.max(0, -dx);
  const xEnd = width - BORDER - Math.max(0, dx);
  const yStart = BORDER + Math.max(0, -dy);
  const yEnd = height - BORDER - Math.max(0, dy);

  let error = 0;
  let count = 0;

  for (let y = yStart; y < yEnd; y += SAMPLE_STEP) {
    const row = y * width;
    const shiftedRow = (y + dy) * width;
    for (let x = xStart; x < xEnd; x += SAMPLE_STEP) {
      error += Math.abs(prevGray[row + x] - currGray[shiftedRow + x + dx]);
      count++;
    }
  }

  return count > 0 ? error / count : Number.POSITIVE_INFINITY;
}

function measureTexture(gray, width, height) {
  let texture = 0;
  let count = 0;

  for (let y = BORDER; y < height - BORDER - 1; y += SAMPLE_STEP * 2) {
    const row = y * width;
    const nextRow = (y + 1) * width;
    for (let x = BORDER; x < width - BORDER - 1; x += SAMPLE_STEP * 2) {
      const idx = row + x;
      texture += Math.abs(gray[idx] - gray[idx + 1]);
      texture += Math.abs(gray[idx] - gray[nextRow + x]);
      count += 2;
    }
  }

  return count > 0 ? texture / count : 0;
}

function refinePeak(left, center, right) {
  const denom = left - (2 * center) + right;
  if (denom === 0) return 0;

  const offset = 0.5 * (left - right) / denom;
  return clamp(offset, -0.5, 0.5);
}

function toGrayscale(imageData) {
  const gray = new Uint8Array(imageData.width * imageData.height);
  const { data } = imageData;

  for (let src = 0, dst = 0; src < data.length; src += 4, dst++) {
    gray[dst] = (data[src] * 77 + data[src + 1] * 150 + data[src + 2] * 29) >> 8;
  }

  return gray;
}

function estimateTranslation(prevGray, currGray, width, height) {
  const texture = measureTexture(currGray, width, height);
  if (texture < MIN_TEXTURE) {
    return { dx: 0, dy: 0, confidence: 0 };
  }

  let bestDx = 0;
  let bestDy = 0;
  let bestScore = Number.POSITIVE_INFINITY;
  let zeroScore = Number.POSITIVE_INFINITY;

  for (let dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; dy++) {
    for (let dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; dx++) {
      const score = scoreShift(prevGray, currGray, width, height, dx, dy);
      if (dx === 0 && dy === 0) zeroScore = score;
      if (score < bestScore) {
        bestScore = score;
        bestDx = dx;
        bestDy = dy;
      }
    }
  }

  const improvement = zeroScore - bestScore;
  if ((bestDx === 0 && bestDy === 0) || improvement < Math.max(MIN_IMPROVEMENT, texture * 0.05)) {
    return { dx: 0, dy: 0, confidence: 0 };
  }

  const leftScore = scoreShift(prevGray, currGray, width, height, bestDx - 1, bestDy);
  const rightScore = scoreShift(prevGray, currGray, width, height, bestDx + 1, bestDy);
  const upScore = scoreShift(prevGray, currGray, width, height, bestDx, bestDy - 1);
  const downScore = scoreShift(prevGray, currGray, width, height, bestDx, bestDy + 1);

  return {
    dx: bestDx + refinePeak(leftScore, bestScore, rightScore),
    dy: bestDy + refinePeak(upScore, bestScore, downScore),
    confidence: improvement,
  };
}

export function getMotionCanvasSize(videoW, videoH) {
  if (!videoW || !videoH) return { width: 0, height: 0 };

  const scale = MOTION_LONGEST / Math.max(videoW, videoH);
  return {
    width: Math.max(32, Math.round(videoW * scale)),
    height: Math.max(32, Math.round(videoH * scale)),
  };
}

export function createMotionState() {
  return {
    ctx: null,
    prevGray: null,
    width: 0,
    height: 0,
    poseX: 0,
    poseY: 0,
    lastUpdateAt: -1,
    lastVideoTime: -1,
  };
}

export function getMotionPose(state) {
  return { x: state.poseX, y: state.poseY };
}

export function updateMotionFromVideo(video, canvas, state, now = performance.now()) {
  if (!video || !canvas || video.readyState < 2 || canvas.width <= 0 || canvas.height <= 0) {
    return getMotionPose(state);
  }

  if (state.prevGray && state.lastUpdateAt >= 0 && now - state.lastUpdateAt < MOTION_MIN_UPDATE_MS) {
    return getMotionPose(state);
  }

  const videoTime = video.currentTime;
  if (videoTime === state.lastVideoTime) {
    return getMotionPose(state);
  }
  state.lastUpdateAt = now;
  state.lastVideoTime = videoTime;

  if (!state.ctx) {
    state.ctx = canvas.getContext('2d', { willReadFrequently: true });
  }

  state.ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const gray = toGrayscale(state.ctx.getImageData(0, 0, canvas.width, canvas.height));

  if (state.width !== canvas.width || state.height !== canvas.height || !state.prevGray) {
    state.width = canvas.width;
    state.height = canvas.height;
    state.prevGray = gray;
    return getMotionPose(state);
  }

  const { dx, dy, confidence } = estimateTranslation(state.prevGray, gray, canvas.width, canvas.height);
  if (confidence > 0) {
    state.poseX += dx;
    state.poseY += dy;
  }

  state.prevGray = gray;
  return getMotionPose(state);
}

export function getCompensationShift(anchorPose, motionState, captureW, captureH) {
  if (!anchorPose || !motionState.width || !motionState.height || !captureW || !captureH) {
    return { x: 0, y: 0 };
  }

  const deltaX = (motionState.poseX - anchorPose.x) * (captureW / motionState.width);
  const deltaY = (motionState.poseY - anchorPose.y) * (captureH / motionState.height);

  return {
    x: clamp(deltaX, -captureW * MAX_COMPENSATION_FRACTION, captureW * MAX_COMPENSATION_FRACTION),
    y: clamp(deltaY, -captureH * MAX_COMPENSATION_FRACTION, captureH * MAX_COMPENSATION_FRACTION),
  };
}

export function compensateDetections(detections, anchorPose, motionState, captureW, captureH) {
  if (!detections.length) {
    return detections;
  }

  const { x: shiftX, y: shiftY } = getCompensationShift(
    anchorPose,
    motionState,
    captureW,
    captureH
  );

  if (shiftX === 0 && shiftY === 0) {
    return detections;
  }

  return detections.map((det) => ({
    ...det,
    bbox: [
      det.bbox[0] + shiftX,
      det.bbox[1] + shiftY,
      det.bbox[2],
      det.bbox[3],
    ],
  }));
}
