import type { ModelConfig } from "@/types/ws";
import { useApp } from "@/context/AppContext";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Collapsible,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import {
  Sheet,
  SheetContent,
} from "@/components/ui/sheet";
import { Settings } from "lucide-react";

function SidebarContent() {
  const {
    config,
    setConfig,
    currentTab,
    captureIntervalMs,
    setCaptureIntervalMs,
  } = useApp();

  return (
    <>
      <h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        Settings
      </h3>
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Thinking Mode</span>
        <Switch
          checked={config.enable_thinking}
          onCheckedChange={(v) => setConfig({ enable_thinking: v })}
        />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Sampling</span>
        <Switch
          checked={config.do_sample}
          onCheckedChange={(v) => setConfig({ do_sample: v })}
        />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Speak responses (TTS)</span>
        <Switch
          checked={config.use_tts}
          onCheckedChange={(v) => {
            setConfig({ use_tts: v });
            try {
              localStorage.setItem("minicpm-tts", v ? "1" : "0");
            } catch {
              /**/
            }
          }}
        />
      </div>
      {[
        { key: "temperature", label: "Temperature", value: config.temperature, min: 0, max: 2, step: 0.05 },
        { key: "top_p", label: "Top P", value: config.top_p, min: 0, max: 1, step: 0.05 },
        { key: "top_k", label: "Top K", value: config.top_k, min: 0, max: 200, step: 1 },
        { key: "repetition_penalty", label: "Rep. Penalty", value: config.repetition_penalty, min: 1, max: 2, step: 0.01 },
        { key: "max_new_tokens", label: "Max New Tokens", value: config.max_new_tokens, min: 64, max: 4096, step: 64 },
      ].map(({ key, label, value, min, max, step }) => (
        <div key={key} className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <Label>{label}</Label>
            <span className="font-semibold text-primary">{value}</span>
          </div>
          <Slider
            min={min}
            max={max}
            step={step}
            value={[value]}
            onValueChange={([v]) =>
              setConfig({ [key as keyof ModelConfig]: v } as Partial<ModelConfig>)
            }
          />
        </div>
      ))}
      <div className="space-y-1">
        <Label className="text-xs text-muted-foreground">System Prompt</Label>
        <Textarea
          placeholder="Optional system prompt…"
          rows={2}
          className="min-h-12 resize-y text-xs"
          value={config.system_prompt}
          onChange={(e) => setConfig({ system_prompt: e.target.value })}
        />
      </div>
      {/* Full duplex only: live settings always relevant when tabs hidden */}
      {currentTab === "live" && (
        <>
          <h3 className="mt-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Live Settings
          </h3>
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <Label>Buffer Size</Label>
              <span className="font-semibold text-primary">{config.max_frames}</span>
            </div>
            <Slider
              min={4}
              max={64}
              step={1}
              value={[config.max_frames]}
              onValueChange={([v]) => setConfig({ max_frames: v })}
            />
          </div>
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <Label>Capture Interval (ms)</Label>
              <span className="font-semibold text-primary">{captureIntervalMs}</span>
            </div>
            <Slider
              min={50}
              max={2000}
              step={50}
              value={[captureIntervalMs]}
              onValueChange={([v]) => setCaptureIntervalMs(v)}
            />
          </div>
        </>
      )}
      {/* deprecated: video mode — Video Settings (UI is full duplex only) */}
      {currentTab === "video" && (
        <>
          <h3 className="mt-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Video Settings
          </h3>
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <Label>Decode FPS</Label>
              <span className="font-semibold text-primary">{config.video_fps}</span>
            </div>
            <Slider
              min={1}
              max={10}
              step={1}
              value={[config.video_fps]}
              onValueChange={([v]) => setConfig({ video_fps: v })}
            />
          </div>
        </>
      )}
    </>
  );
}

export function Sidebar() {
  const { mobileSettingsOpen, setMobileSettingsOpen, sidebarOpen, setSidebarOpen } = useApp();

  return (
    <>
      {/* Desktop: collapsible inline sidebar (trigger is in TopBar) */}
      <div className="relative hidden shrink-0 md:flex">
        <Collapsible open={sidebarOpen} onOpenChange={setSidebarOpen} className="flex">
          <CollapsibleContent
            className="flex data-[state=closed]:hidden"
            style={{ width: sidebarOpen ? 280 : 0, minWidth: sidebarOpen ? 280 : 0 }}
          >
            <aside
              className="flex w-[280px] flex-col gap-3.5 overflow-y-auto border-r border-border bg-card p-4"
              aria-label="Settings"
            >
              <SidebarContent />
            </aside>
          </CollapsibleContent>
        </Collapsible>
      </div>

      {/* Mobile: sheet overlay */}
      <Sheet open={mobileSettingsOpen} onOpenChange={setMobileSettingsOpen}>
        <SheetContent side="left" className="w-[280px] max-w-[85vw] overflow-y-auto p-4 sm:max-w-sm">
          <div className="flex flex-col gap-3.5 pt-6" aria-label="Settings">
            <SidebarContent />
          </div>
        </SheetContent>
      </Sheet>
    </>
  );
}

export function SidebarMobileTrigger() {
  const { setMobileSettingsOpen } = useApp();
  return (
    <Button
      variant="ghost"
      size="icon"
      className="md:hidden"
      onClick={() => setMobileSettingsOpen(true)}
      aria-label="Open settings"
    >
      <Settings className="size-5" />
    </Button>
  );
}
