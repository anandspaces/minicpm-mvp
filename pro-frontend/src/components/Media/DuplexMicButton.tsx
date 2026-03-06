import { useApp } from "@/context/AppContext";
import { Mic } from "lucide-react";
import { cn } from "@/lib/utils";

export function DuplexMicButton() {
  const { mediaStream, duplexActive, startDuplex, stopDuplex, streamError } = useApp();
  const isActive = Boolean(mediaStream);

  const handleClick = () => {
    if (isActive) {
      stopDuplex();
    } else {
      startDuplex();
    }
  };

  return (
    <div className="flex shrink-0 flex-col items-center justify-center gap-2 pb-6 pt-4">
      {streamError && (
        <p className="text-xs text-destructive" role="alert">
          {streamError}
        </p>
      )}
      <button
        type="button"
        onClick={handleClick}
        className={cn(
          "relative flex h-16 w-16 items-center justify-center rounded-full border-2 transition-colors",
          "bg-primary text-primary-foreground hover:bg-primary/90 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
          "disabled:pointer-events-none disabled:opacity-50"
        )}
        aria-label={isActive ? "Stop live stream" : "Start live stream"}
      >
        {/* Pulsing ring when streaming */}
        {isActive && (
          <span
            className="absolute inset-0 rounded-full border-2 border-primary/60 animate-ping opacity-30"
            aria-hidden
          />
        )}
        {isActive && duplexActive && (
          <span
            className="absolute -inset-1 rounded-full border-2 border-green-500/50 animate-pulse"
            aria-hidden
          />
        )}
        <Mic className="relative size-8 shrink-0" />
      </button>
    </div>
  );
}
