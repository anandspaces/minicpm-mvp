import { useApp } from "@/context/AppContext";
import { cn } from "@/lib/utils";

export function TopBar() {
  const { wsStatus } = useApp();
  const connected = wsStatus === "open";
  const statusText =
    wsStatus === "open"
      ? "Connected"
      : wsStatus === "connecting"
        ? "Connecting…"
        : "Disconnected — reconnecting…";

  return (
    <header className="flex shrink-0 items-center justify-between border-b border-border bg-card px-5 py-2.5">
      <h1 className="text-base font-semibold">MiniCPM-o 4.5</h1>
      <div
        className="flex items-center gap-1.5 text-xs text-muted-foreground"
        role="status"
        aria-live="polite"
      >
        <span
          className={cn(
            "size-2 rounded-full",
            connected ? "bg-green-500" : "bg-destructive"
          )}
          aria-hidden
        />
        <span>{statusText}</span>
      </div>
    </header>
  );
}
