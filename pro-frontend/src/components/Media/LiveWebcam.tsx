/**
 * Full duplex only. Video is driven by context (mediaStream); start/stop is via DuplexMicButton.
 */
import { useCallback, useEffect, useRef } from "react";
import { useApp } from "@/context/AppContext";
import { captureAudioChunk, DUPLEX_CHUNK_MS } from "@/utils/duplexAudio";

export function LiveWebcam() {
  const { send, mediaStream, duplexActive, stopDuplex } = useApp();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!videoRef.current) return;
    videoRef.current.srcObject = mediaStream;
  }, [mediaStream]);

  const getFrameB64 = useCallback((): string | null => {
    const video = videoRef.current;
    if (!video?.videoWidth) return null;
    let canvas = canvasRef.current;
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvasRef.current = canvas;
    }
    const scale = Math.min(1, 640 / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
    return dataUrl.includes(",") ? dataUrl.split(",")[1] ?? null : null;
  }, []);

  // Duplex chunk loop when stream and duplex are active
  useEffect(() => {
    if (!mediaStream || !duplexActive) return;
    const id = setInterval(() => {
      const frameB64 = getFrameB64();
      captureAudioChunk(mediaStream)
        .then((audioB64) => {
          send({
            action: "duplex_chunk",
            audio_b64: audioB64,
            ...(frameB64 ? { frame_b64: frameB64 } : {}),
          });
        })
        .catch(() => {});
    }, DUPLEX_CHUNK_MS);
    return () => clearInterval(id);
  }, [mediaStream, duplexActive, send, getFrameB64]);

  // On unmount, end duplex session if active (no-op if no stream)
  useEffect(() => () => stopDuplex(), [stopDuplex]);

  return (
    <div className="relative flex min-h-0 min-w-0 flex-1 flex-col items-center justify-center bg-black/90">
      {mediaStream ? (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="max-h-full max-w-full object-contain"
        />
      ) : (
        <p className="text-sm text-muted-foreground">Tap mic to start</p>
      )}
    </div>
  );
}
