import { useCallback, useRef, useState } from "react";
import { useApp } from "@/context/AppContext";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export function VideoUpload() {
  const { send, config, uploadStatus, uploadMessage } = useApp();
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File | null) => {
      if (!file) return;
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = reader.result as string;
        const b64 = dataUrl.includes(",") ? dataUrl.split(",")[1] : dataUrl;
        if (b64) send({ action: "upload_video", data: b64, fps: config.video_fps });
      };
      reader.readAsDataURL(file);
    },
    [send, config.video_fps, previewUrl]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.currentTarget.classList.remove("border-primary");
      handleFile(e.dataTransfer.files[0] ?? null);
    },
    [handleFile]
  );
  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.add("border-primary");
  }, []);
  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.currentTarget.classList.remove("border-primary");
  }, []);

  return (
    <div className="flex flex-col gap-2">
      <Card
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center border-2 border-dashed p-7 text-center text-sm text-muted-foreground transition-colors hover:border-primary/50",
          uploadStatus === "uploading" && "opacity-70"
        )}
        onClick={() => inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => {
            handleFile(e.target.files?.[0] ?? null);
            e.target.value = "";
          }}
        />
        <span>
          {uploadStatus === "uploading"
            ? "Uploading video…"
            : "Drop a video file here or click to select"}
        </span>
      </Card>
      {previewUrl && (
        <video
          src={previewUrl}
          controls
          className="max-h-40 w-full rounded-lg bg-black object-contain"
        />
      )}
      {(fileName || uploadMessage) && (
        <p className="text-xs text-muted-foreground">
          {fileName}
          {uploadMessage && ` — ${uploadMessage}`}
        </p>
      )}
    </div>
  );
}
