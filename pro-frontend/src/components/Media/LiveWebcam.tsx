import { useCallback, useEffect, useRef, useState } from "react";
import { useApp } from "@/context/AppContext";
import { Button } from "@/components/ui/button";

export function LiveWebcam() {
  const { send, captureIntervalMs } = useApp();
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [status, setStatus] = useState("Camera off");
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video?.videoWidth || !stream) return;
    let canvas = canvasRef.current;
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvasRef.current = canvas;
    }
    const scale = Math.min(1, 640 / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
    const b64 = dataUrl.includes(",") ? dataUrl.split(",")[1] : "";
    if (b64) send({ action: "frame", data: b64 });
  }, [send, stream]);

  const startCam = useCallback(async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      setStream(s);
      if (videoRef.current) videoRef.current.srcObject = s;
      setStatus("Camera active — streaming frames");
      intervalRef.current = setInterval(captureFrame, captureIntervalMs);
    } catch (err) {
      setStatus(`Camera error: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [captureIntervalMs, captureFrame]);

  const stopCam = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      setStream(null);
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("Camera off");
  }, [stream]);

  useEffect(() => {
    if (!stream || !intervalRef.current) return;
    clearInterval(intervalRef.current);
    intervalRef.current = setInterval(captureFrame, captureIntervalMs);
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [captureIntervalMs, stream, captureFrame]);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [stream]);

  return (
    <div className="flex gap-4">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="h-[180px] w-[240px] rounded-lg bg-black object-cover"
      />
      <div className="flex flex-col gap-2">
        <Button
          variant="default"
          className="bg-green-600 hover:bg-green-700"
          onClick={startCam}
          style={{ display: stream ? "none" : undefined }}
        >
          Start Camera
        </Button>
        <Button variant="destructive" onClick={stopCam} style={{ display: stream ? undefined : "none" }}>
          Stop Camera
        </Button>
        <p className="text-xs text-muted-foreground">{status}</p>
      </div>
    </div>
  );
}
