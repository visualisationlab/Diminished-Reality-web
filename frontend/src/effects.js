export const EFFECT = { NONE: 0, BLUR: 1, OVERLAY: 2, DESATURATE: 3 };
export const OUTLINE = { OFF: 0, HEALTHY: 1, ALL: 2 };

const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
const BLUR_PAD_MIN = 10;
const BLUR_PAD_MAX = 28;
const BLUR_PAD_RATIO = 0.14;
const BLUR_PAD_STRENGTH_RATIO = 0.5;

function createWorkingCanvas(width, height) {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(width, height);
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

// --- Coordinate transform: capture coords → display coords (object-fit: cover) ---

function getCoverCrop(videoW, videoH, displayW, displayH) {
  const displayAR = displayW / displayH;
  const videoAR = videoW / videoH;

  if (videoAR > displayAR) {
    return {
      sx: (videoW - videoH * displayAR) / 2,
      sy: 0,
      sw: videoH * displayAR,
      sh: videoH,
    };
  }

  return {
    sx: 0,
    sy: (videoH - videoW / displayAR) / 2,
    sw: videoW,
    sh: videoW / displayAR,
  };
}

function captureToDisplay(bbox, captureW, captureH, videoW, videoH, displayW, displayH) {
  // 1. Capture → native video coords
  const scaleX = videoW / captureW;
  const scaleY = videoH / captureH;
  let [x, y, w, h] = bbox;
  x *= scaleX;
  y *= scaleY;
  w *= scaleX;
  h *= scaleY;

  // 2. Compute object-fit: cover crop rect in native video coords
  const { sx, sy, sw, sh } = getCoverCrop(videoW, videoH, displayW, displayH);

  // 3. Native → display
  const dx = (x - sx) * (displayW / sw);
  const dy = (y - sy) * (displayH / sh);
  const dw = w * (displayW / sw);
  const dh = h * (displayH / sh);

  return [dx, dy, dw, dh];
}

export function captureDeltaToDisplay(deltaX, deltaY, captureW, captureH, videoW, videoH, displayW, displayH) {
  if (!captureW || !captureH || !videoW || !videoH || !displayW || !displayH) {
    return [0, 0];
  }

  const scaleX = videoW / captureW;
  const scaleY = videoH / captureH;
  const { sw, sh } = getCoverCrop(videoW, videoH, displayW, displayH);

  return [
    deltaX * scaleX * (displayW / sw),
    deltaY * scaleY * (displayH / sh),
  ];
}

export function expandBlurRect(x, y, w, h, strength, boundsW, boundsH) {
  if (w <= 0 || h <= 0 || boundsW <= 0 || boundsH <= 0) {
    return { x, y, w, h };
  }

  const padX = Math.min(
    BLUR_PAD_MAX,
    Math.max(BLUR_PAD_MIN, strength * BLUR_PAD_STRENGTH_RATIO, w * BLUR_PAD_RATIO)
  );
  const padY = Math.min(
    BLUR_PAD_MAX,
    Math.max(BLUR_PAD_MIN, strength * BLUR_PAD_STRENGTH_RATIO, h * BLUR_PAD_RATIO)
  );

  const ex = Math.max(0, x - padX);
  const ey = Math.max(0, y - padY);
  const ex2 = Math.min(boundsW, x + w + padX);
  const ey2 = Math.min(boundsH, y + h + padY);

  return {
    x: ex,
    y: ey,
    w: Math.max(0, ex2 - ex),
    h: Math.max(0, ey2 - ey),
  };
}

// --- iOS manual pixel effects ---

function manualBlur(imageData, radius) {
  const { data, width, height } = imageData;
  const orig = new Uint8ClampedArray(data);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0, g = 0, b = 0, a = 0, n = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const i = (ny * width + nx) * 4;
            r += orig[i]; g += orig[i + 1]; b += orig[i + 2]; a += orig[i + 3];
            n++;
          }
        }
      }
      const i = (y * width + x) * 4;
      data[i] = r / n; data[i + 1] = g / n; data[i + 2] = b / n; data[i + 3] = a / n;
    }
  }
}

function manualDesaturate(imageData, strength) {
  const { data } = imageData;
  for (let i = 0; i < data.length; i += 4) {
    const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    data[i] += (lum - data[i]) * strength;
    data[i + 1] += (lum - data[i + 1]) * strength;
    data[i + 2] += (lum - data[i + 2]) * strength;
  }
}

// --- Effect applicators ---

function displayToVideoRect(x, y, w, h, videoW, videoH, displayW, displayH) {
  const { sx, sy, sw, sh } = getCoverCrop(videoW, videoH, displayW, displayH);

  return [
    sx + (x * sw) / displayW,
    sy + (y * sh) / displayH,
    (w * sw) / displayW,
    (h * sh) / displayH,
  ];
}

function applyBlur(ctx, video, x, y, w, h, strength) {
  if (strength <= 0) return;
  const expanded = expandBlurRect(x, y, w, h, strength, ctx.canvas.width, ctx.canvas.height);
  const ix = Math.max(0, Math.floor(expanded.x));
  const iy = Math.max(0, Math.floor(expanded.y));
  const iw = Math.min(ctx.canvas.width - ix, Math.ceil(expanded.w));
  const ih = Math.min(ctx.canvas.height - iy, Math.ceil(expanded.h));
  if (iw <= 0 || ih <= 0) return;

  // Sample past the detection bounds so the blur stays full-strength up to the edge.
  const pad = Math.max(2, Math.ceil(strength * 2));
  const ex = Math.max(0, ix - pad);
  const ey = Math.max(0, iy - pad);
  const ex2 = Math.min(ctx.canvas.width, ix + iw + pad);
  const ey2 = Math.min(ctx.canvas.height, iy + ih + pad);
  const ew = ex2 - ex;
  const eh = ey2 - ey;
  const [sx, sy, sw, sh] = displayToVideoRect(
    ex, ey, ew, eh, video.videoWidth, video.videoHeight, ctx.canvas.width, ctx.canvas.height
  );
  const off = createWorkingCanvas(ew, eh);
  const oc = off.getContext('2d');

  if (isIOS) {
    oc.drawImage(video, sx, sy, sw, sh, 0, 0, ew, eh);
    const imgData = oc.getImageData(0, 0, ew, eh);
    manualBlur(imgData, Math.min(strength, 8));
    oc.putImageData(imgData, 0, 0);
  } else {
    oc.filter = `blur(${strength}px)`;
    oc.drawImage(video, sx, sy, sw, sh, 0, 0, ew, eh);
  }

  ctx.drawImage(off, ix - ex, iy - ey, iw, ih, ix, iy, iw, ih);
}

function applyOverlay(ctx, x, y, w, h, opacity) {
  if (opacity <= 0) return;
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.fillStyle = 'white';
  ctx.fillRect(x, y, w, h);
  ctx.restore();
}

function applyDesaturate(ctx, video, x, y, w, h, strength) {
  if (strength <= 0) return;
  const ix = Math.max(0, Math.round(x));
  const iy = Math.max(0, Math.round(y));
  const iw = Math.round(w);
  const ih = Math.round(h);
  if (iw <= 0 || ih <= 0) return;

  if (isIOS) {
    const off = new OffscreenCanvas(iw, ih);
    const oc = off.getContext('2d');
    oc.drawImage(ctx.canvas, ix, iy, iw, ih, 0, 0, iw, ih);
    const imgData = oc.getImageData(0, 0, iw, ih);
    manualDesaturate(imgData, strength);
    oc.putImageData(imgData, 0, 0);
    ctx.drawImage(off, ix, iy);
  } else {
    ctx.save();
    ctx.beginPath();
    ctx.rect(ix, iy, iw, ih);
    ctx.clip();
    ctx.filter = `saturate(${1 - strength})`;
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight,
                  0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.restore();
  }
}

function drawOutline(ctx, x, y, w, h, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
}

// --- Strength maps ---

export const EFFECT_STRENGTH = {
  [EFFECT.NONE]: 0,
  [EFFECT.BLUR]: 20,
  [EFFECT.OVERLAY]: 1,
  [EFFECT.DESATURATE]: 1,
};

// --- Main render function ---

export function prepareRenderDetections(
  detections, classOverrides, captureW, captureH, videoW, videoH, displayW, displayH
) {
  if (!videoW || !videoH || !captureW || !captureH || !displayW || !displayH) {
    return [];
  }

  return detections.map((det) => {
    const isHealthy = classOverrides[det.class] !== undefined
      ? classOverrides[det.class]
      : det.in_schijf_van_vijf;
    const [x, y, w, h] = captureToDisplay(
      det.bbox, captureW, captureH, videoW, videoH, displayW, displayH
    );

    return {
      ...det,
      isHealthy,
      opacity: det.opacity ?? 1,
      x,
      y,
      w,
      h,
    };
  });
}

export function renderEffects(canvas, video, detections, effectType, outlineMode, outlineColor) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const det of detections) {
    const { isHealthy, opacity, x, y, w, h } = det;

    // Outline
    if (outlineMode === OUTLINE.ALL || (outlineMode === OUTLINE.HEALTHY && isHealthy)) {
      const oColor = outlineColor === 'health_based'
        ? (isHealthy ? '#22c55e' : '#ef4444')
        : outlineColor;
      ctx.globalAlpha = opacity;
      drawOutline(ctx, x, y, w, h, oColor);
      ctx.globalAlpha = 1;
    }

    // Diminish effect (only for unhealthy)
    if (!isHealthy) {
      const strength = EFFECT_STRENGTH[effectType] * opacity;
      switch (effectType) {
        case EFFECT.BLUR:
          applyBlur(ctx, video, x, y, w, h, strength);
          break;
        case EFFECT.OVERLAY:
          applyOverlay(ctx, x, y, w, h, strength);
          break;
        case EFFECT.DESATURATE:
          applyDesaturate(ctx, video, x, y, w, h, strength);
          break;
      }
    }
  }
}
