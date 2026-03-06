import { useCallback, useRef, useState } from "react";
import { useApp } from "@/context/AppContext";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export function PhotoDropzone() {
  const { send, uploadStatus, uploadMessage } = useApp();
  const inputRef = useRef<HTMLInputElement>(null);
  const [thumbUrls, setThumbUrls] = useState<string[]>([]);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files?.length) return;
      const fileArray = Array.from(files);
      const dataUrlPromises = fileArray.map(
        (f) =>
          new Promise<string>((resolve, reject) => {
            const r = new FileReader();
            r.onload = () => resolve((r.result as string) ?? "");
            r.onerror = () => reject(r.error);
            r.readAsDataURL(f);
          })
      );
      const b64Promises = fileArray.map(
        (f) =>
          new Promise<string>((resolve, reject) => {
            const r = new FileReader();
            r.onload = () => {
              const dataUrl = r.result as string;
              const b64 = dataUrl.includes(",") ? dataUrl.split(",")[1] : dataUrl;
              resolve(b64 ?? "");
            };
            r.onerror = () => reject(r.error);
            r.readAsDataURL(f);
          })
      );
      Promise.all(dataUrlPromises).then((urls) => setThumbUrls(urls));
      Promise.all(b64Promises).then((b64List) => {
        send({ action: "upload_images", images: b64List.filter(Boolean) });
      });
    },
    [send]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.currentTarget.classList.remove("border-primary");
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );
  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.add("border-primary");
  }, []);
  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.currentTarget.classList.remove("border-primary");
  }, []);

  return (
    <div className="space-y-2">
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
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => {
            handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <span>
          {uploadStatus === "uploading"
            ? "Uploading…"
            : "Drop images here or click to select"}
        </span>
      </Card>
      {thumbUrls.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {thumbUrls.map((url, i) => (
            <img
              key={i}
              src={url}
              alt=""
              className="size-16 rounded-md border border-border object-cover"
            />
          ))}
        </div>
      )}
      {uploadMessage && uploadStatus === "done" && (
        <p className="text-xs text-muted-foreground">{uploadMessage}</p>
      )}
    </div>
  );
}
