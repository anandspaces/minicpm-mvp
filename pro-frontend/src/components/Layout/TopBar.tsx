import { useApp } from "@/context/AppContext";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Menu } from "lucide-react";
import { SidebarMobileTrigger } from "./Sidebar";

export function TopBar() {
  const { wsStatus, sidebarOpen, setSidebarOpen } = useApp();
  const connected = wsStatus === "open";
  const statusText =
    wsStatus === "open"
      ? "Connected"
      : wsStatus === "connecting"
        ? "Connecting…"
        : "Disconnected — reconnecting…";

  return (
    <header className="flex shrink-0 items-center justify-between gap-2 border-b border-border bg-card px-3 py-2 sm:px-5 sm:py-2.5">
      <div className="flex min-w-0 flex-1 items-center gap-2">
        <SidebarMobileTrigger />
        <Button
          variant="outline"
          size="icon"
          className="hidden h-8 w-8 md:flex"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          aria-label={sidebarOpen ? "Hide settings" : "Show settings"}
        >
          <Menu className={cn("size-4 transition-transform", sidebarOpen && "rotate-180")} />
        </Button>
        <h1 className="truncate text-base font-semibold max-w-[120px] sm:max-w-none">MiniCPM-o 4.5</h1>
      </div>
      <div
        className="flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground"
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
        <span className="truncate">{statusText}</span>
      </div>
    </header>
  );
}
