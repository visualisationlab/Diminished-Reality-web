import { useRef, useEffect, useState, useCallback } from 'react';
import { captureDeltaToDisplay, prepareRenderDetections, renderEffects, EFFECT, EFFECT_STRENGTH, OUTLINE } from './effects';
import { compensateDetections, createMotionState, getCompensationShift, getMotionCanvasSize, getMotionPose, updateMotionFromVideo } from './motion';
import Settings from './Settings';

const CAPTURE_LONGEST = 640;
const SEND_TIMEOUT = 30;
const MIN_SEND_INTERVAL = 10;
const JPEG_QUALITY = 0.7;

function supportsNativeBlur() {
  return typeof CSS !== 'undefined' && typeof CSS.supports === 'function' && (
    CSS.supports('backdrop-filter', 'blur(1px)') ||
    CSS.supports('-webkit-backdrop-filter', 'blur(1px)')
  );
}

const nativeBlurCache = new WeakMap();

function boxesEqual(a, b) {
  return Math.abs(a.x - b.x) < 0.25 &&
    Math.abs(a.y - b.y) < 0.25 &&
    Math.abs(a.w - b.w) < 0.25 &&
    Math.abs(a.h - b.h) < 0.25 &&
    Math.abs(a.opacity - b.opacity) < 0.01;
}

function syncNativeBlurLayer(layer, detections, offsetX = 0, offsetY = 0) {
  if (!layer) return;
  const cached = nativeBlurCache.get(layer) ?? { boxes: [], offsetX: 0, offsetY: 0 };

  while (layer.childElementCount > detections.length) {
    layer.lastElementChild.remove();
  }

  while (layer.childElementCount < detections.length) {
    const box = document.createElement('div');
    box.className = 'native-blur-box';
    layer.appendChild(box);
  }

  layer.style.display = detections.length ? 'block' : 'none';
  if (!detections.length) {
    layer.style.transform = '';
    nativeBlurCache.set(layer, { boxes: [], offsetX: 0, offsetY: 0 });
    return;
  }

  if (Math.abs(cached.offsetX - offsetX) >= 0.25 || Math.abs(cached.offsetY - offsetY) >= 0.25) {
    layer.style.transform = `translate3d(${offsetX}px, ${offsetY}px, 0)`;
  }

  let needsBoxSync = cached.boxes.length !== detections.length;
  if (!needsBoxSync) {
    for (let i = 0; i < detections.length; i++) {
      if (!boxesEqual(cached.boxes[i], detections[i])) {
        needsBoxSync = true;
        break;
      }
    }
  }

  if (!needsBoxSync) {
    nativeBlurCache.set(layer, { boxes: cached.boxes, offsetX, offsetY });
    return;
  }

  const nextBoxes = [];
  for (let i = 0; i < detections.length; i++) {
    const det = detections[i];
    const box = layer.children[i];
    const blur = `blur(${Math.max(0, EFFECT_STRENGTH[EFFECT.BLUR] * det.opacity)}px)`;

    box.style.transform = `translate3d(${det.x}px, ${det.y}px, 0)`;
    box.style.width = `${det.w}px`;
    box.style.height = `${det.h}px`;
    box.style.opacity = `${det.opacity}`;
    box.style.backdropFilter = blur;
    box.style.webkitBackdropFilter = blur;

    nextBoxes.push({
      x: det.x,
      y: det.y,
      w: det.w,
      h: det.h,
      opacity: det.opacity,
    });
  }

  nativeBlurCache.set(layer, { boxes: nextBoxes, offsetX, offsetY });
}

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const motionCanvasRef = useRef(null);
  const containerRef = useRef(null);
  const blurLayerRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [effect, setEffect] = useState(EFFECT.BLUR);
  const [outlineMode, setOutlineMode] = useState(OUTLINE.OFF);
  const [outlineColor, setOutlineColor] = useState('health_based');
  const [classOverrides, setClassOverrides] = useState({});

  // Refs for rAF-accessible state
  const settingsRef = useRef({ effect, outlineMode, outlineColor, classOverrides });
  const detectionsRef = useRef({ items: [], anchorPose: null });
  const captureSize = useRef({ w: 0, h: 0 });
  const wsRef = useRef(null);
  const pendingRef = useRef(false);
  const sendTimerRef = useRef(null);
  const requestIdRef = useRef(0);
  const lastSendAtRef = useRef(0);
  const nativeBlurSupportedRef = useRef(supportsNativeBlur());
  const motionStateRef = useRef(createMotionState());
  const sentPoseByRequestRef = useRef(new Map());

  // Sync settings to ref
  useEffect(() => {
    settingsRef.current = { effect, outlineMode, outlineColor, classOverrides };
  }, [effect, outlineMode, outlineColor, classOverrides]);

  // Send overrides to backend when user changes them
  const skipNextSyncRef = useRef(false);
  const mountedRef = useRef(false);
  useEffect(() => {
    if (!mountedRef.current) { mountedRef.current = true; return; }
    if (skipNextSyncRef.current) { skipNextSyncRef.current = false; return; }
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'save_settings', overrides: classOverrides }));
    }
  }, [classOverrides]);

  // --- Camera ---
  useEffect(() => {
    let stream = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
          audio: false,
        });
        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          video.play();
        }
      } catch (e) {
        console.error('Camera error:', e);
      }
    })();
    return () => stream?.getTracks().forEach(t => t.stop());
  }, []);

  // Size capture canvas once video dimensions are known
  const onVideoReady = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    const { videoWidth: vw, videoHeight: vh } = video;
    if (!vw || !vh) return;
    const scale = CAPTURE_LONGEST / Math.max(vw, vh);
    const cw = Math.round(vw * scale);
    const ch = Math.round(vh * scale);
    captureSize.current = { w: cw, h: ch };
    if (captureCanvasRef.current) {
      captureCanvasRef.current.width = cw;
      captureCanvasRef.current.height = ch;
    }
    const motionSize = getMotionCanvasSize(vw, vh);
    if (motionCanvasRef.current) {
      motionCanvasRef.current.width = motionSize.width;
      motionCanvasRef.current.height = motionSize.height;
    }
  }, []);

  // --- WebSocket ---
  useEffect(() => {
    let ws;
    let reconnectTimer;

    function connect() {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WS open');
        setConnected(true);
        // Request saved overrides from backend
        ws.send(JSON.stringify({ type: 'get_settings' }));
      };
      ws.onclose = (event) => {
        console.log('WS close', { code: event.code, reason: event.reason });
        setConnected(false);
        detectionsRef.current = { items: [], anchorPose: null };
        sentPoseByRequestRef.current.clear();
        wsRef.current = null;
        reconnectTimer = setTimeout(connect, 2000);
      };
      ws.onerror = (event) => {
        console.error('WS error', event);
        ws.close();
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === 'settings') {
            // Load saved overrides from backend without echoing back
            if (data.overrides) {
              skipNextSyncRef.current = true;
              setClassOverrides(data.overrides);
            }
            return;
          }
          const requestId = data.requestId;
          const anchorPose = requestId !== undefined
            ? sentPoseByRequestRef.current.get(requestId) ?? getMotionPose(motionStateRef.current)
            : getMotionPose(motionStateRef.current);
          detectionsRef.current = { items: data.detections || [], anchorPose };

          if (requestId !== undefined) {
            for (const key of Array.from(sentPoseByRequestRef.current.keys())) {
              if (key <= requestId) {
                sentPoseByRequestRef.current.delete(key);
              }
            }
          }
        } catch (e) {
          console.error('WS message error:', e);
        }

        // Unblock next send
        pendingRef.current = false;
        clearTimeout(sendTimerRef.current);
      };
    }

    connect();
    return () => {
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, []);

  // --- rAF loop: capture + render ---
  useEffect(() => {
    let rafId;

    const loop = () => {
      const now = performance.now();
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const capCanvas = captureCanvasRef.current;
      const motionCanvas = motionCanvasRef.current;
      const blurLayer = blurLayerRef.current;

      if (video && canvas && video.readyState >= 2) {
        // Match canvas to display size
        const rect = video.getBoundingClientRect();
        if (canvas.width !== rect.width) canvas.width = rect.width;
        if (canvas.height !== rect.height) canvas.height = rect.height;
        const shouldTrackMotion = wsRef.current?.readyState === WebSocket.OPEN ||
          pendingRef.current ||
          detectionsRef.current.items.length > 0;
        if (shouldTrackMotion) {
          updateMotionFromVideo(video, motionCanvas, motionStateRef.current, now);
        }

        // Send frame if WS is ready and not waiting for response
        if (wsRef.current?.readyState === WebSocket.OPEN && !pendingRef.current && capCanvas) {
          const { w: cw, h: ch } = captureSize.current;
          if (cw > 0 && ch > 0 && now - lastSendAtRef.current >= MIN_SEND_INTERVAL) {
            pendingRef.current = true;
            lastSendAtRef.current = now;

            const cCtx = capCanvas.getContext('2d');
            cCtx.drawImage(video, 0, 0, cw, ch);
            requestIdRef.current = (requestIdRef.current + 1) & 0xFFFFFFFF;
            const id = requestIdRef.current;
            sentPoseByRequestRef.current.set(id, getMotionPose(motionStateRef.current));
            while (sentPoseByRequestRef.current.size > 8) {
              const oldest = sentPoseByRequestRef.current.keys().next().value;
              sentPoseByRequestRef.current.delete(oldest);
            }

            capCanvas.toBlob((blob) => {
              if (!blob || wsRef.current?.readyState !== WebSocket.OPEN) {
                pendingRef.current = false;
                sentPoseByRequestRef.current.delete(id);
                return;
              }

              blob.arrayBuffer().then((buf) => {
                const msg = new ArrayBuffer(4 + buf.byteLength);
                new DataView(msg).setUint32(0, id, false);
                new Uint8Array(msg, 4).set(new Uint8Array(buf));
                wsRef.current?.send(msg);

                // Timeout: unblock if server is slow
                sendTimerRef.current = setTimeout(() => { pendingRef.current = false; }, SEND_TIMEOUT);
              });
            }, 'image/jpeg', JPEG_QUALITY);
          }
        }

        // Render effects
        const s = settingsRef.current;
        const shift = getCompensationShift(
          detectionsRef.current.anchorPose,
          motionStateRef.current,
          captureSize.current.w,
          captureSize.current.h
        );
        const [blurOffsetX, blurOffsetY] = captureDeltaToDisplay(
          shift.x,
          shift.y,
          captureSize.current.w,
          captureSize.current.h,
          video.videoWidth,
          video.videoHeight,
          rect.width,
          rect.height
        );
        const baseRenderDetections = prepareRenderDetections(
          detectionsRef.current.items,
          s.classOverrides,
          captureSize.current.w,
          captureSize.current.h,
          video.videoWidth,
          video.videoHeight,
          rect.width,
          rect.height
        );
        const compensatedDetections = compensateDetections(
          detectionsRef.current.items,
          detectionsRef.current.anchorPose,
          motionStateRef.current,
          captureSize.current.w,
          captureSize.current.h
        );
        const renderDetections = prepareRenderDetections(
          compensatedDetections,
          s.classOverrides,
          captureSize.current.w,
          captureSize.current.h,
          video.videoWidth,
          video.videoHeight,
          rect.width,
          rect.height
        );
        const useNativeBlur = nativeBlurSupportedRef.current && s.effect === EFFECT.BLUR;

        syncNativeBlurLayer(
          blurLayer,
          useNativeBlur
            ? baseRenderDetections.filter((det) => !det.isHealthy && det.opacity > 0 && det.w > 0 && det.h > 0)
            : [],
          useNativeBlur ? blurOffsetX : 0,
          useNativeBlur ? blurOffsetY : 0
        );

        renderEffects(
          canvas,
          video,
          renderDetections,
          useNativeBlur ? EFFECT.NONE : s.effect,
          s.outlineMode,
          s.outlineColor
        );
      }

      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, []);

  // --- Fullscreen ---
  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else if (el.webkitRequestFullscreen) {
      el.webkitRequestFullscreen();
    } else {
      el.requestFullscreen();
    }
  }, []);

  return (
    <div className="app" ref={containerRef}>
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        onLoadedMetadata={onVideoReady}
      />
      <div ref={blurLayerRef} className="native-blur-layer" />
      <canvas ref={canvasRef} className="overlay-canvas" />
      <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
      <canvas ref={motionCanvasRef} style={{ display: 'none' }} />

      <div className="controls">
        <button className="ctrl-btn" onClick={toggleFullscreen} aria-label="Fullscreen">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 3 21 3 21 9" /><polyline points="9 21 3 21 3 15" />
            <line x1="21" y1="3" x2="14" y2="10" /><line x1="3" y1="21" x2="10" y2="14" />
          </svg>
        </button>
        <button className="ctrl-btn" onClick={() => setSettingsOpen(true)} aria-label="Settings">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
      </div>

      <div className={`status-dot ${connected ? 'connected' : 'disconnected'}`} />

      <Settings
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        effect={effect}
        setEffect={setEffect}
        outlineMode={outlineMode}
        setOutlineMode={setOutlineMode}
        outlineColor={outlineColor}
        setOutlineColor={setOutlineColor}
        classOverrides={classOverrides}
        setClassOverrides={setClassOverrides}
      />
    </div>
  );
}

export default App;
