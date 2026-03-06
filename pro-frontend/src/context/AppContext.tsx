import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useWebSocket } from "@/hooks/useWebSocket";
import type {
  ChatMessage,
  ModelConfig,
  ServerEvent,
  Mode,
} from "@/types/ws";

function getInitialConfig(): ModelConfig {
  const base: ModelConfig = {
    enable_thinking: false,
    do_sample: true,
    temperature: 0.7,
    top_p: 0.8,
    top_k: 100,
    repetition_penalty: 1.05,
    max_new_tokens: 2048,
    system_prompt: "",
    max_frames: 16,
    video_fps: 3,
    use_tts: false,
  };
  try {
    if (localStorage.getItem("minicpm-tts") !== "0") base.use_tts = true;
  } catch {
    /**/
  }
  return base;
}

function nextId() {
  return crypto.randomUUID();
}

type UploadStatus = "idle" | "uploading" | "done";

type AppState = {
  config: ModelConfig;
  messages: ChatMessage[];
  isStreaming: boolean;
  uploadStatus: UploadStatus;
  uploadMessage: string | null;
  currentTab: Mode;
  lastChatAudio: { audio_b64: string; sample_rate: number } | null;
  captureIntervalMs: number;
  lastServerError: string | null;
  duplexActive: boolean;
  mediaStream: MediaStream | null;
  streamError: string | null;
};

type AppContextValue = AppState & {
  wsStatus: "connecting" | "open" | "closed" | "error";
  lastError: string | null;
  mobileSettingsOpen: boolean;
  setMobileSettingsOpen: (open: boolean) => void;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  send: (action: import("@/types/ws").ClientAction) => void;
  setConfig: (patch: Partial<ModelConfig>) => void;
  setCurrentTab: (tab: Mode) => void;
  addSystemMessage: (text: string) => void;
  addUserMessage: (content: string) => void;
  clearLastChatAudio: () => void;
  clearLastServerError: () => void;
  setCaptureIntervalMs: (ms: number) => void;
  startDuplex: () => Promise<void>;
  stopDuplex: () => void;
};

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AppState>({
    config: getInitialConfig(),
    messages: [],
    isStreaming: false,
    uploadStatus: "idle",
    uploadMessage: null,
    currentTab: "live", // full duplex only; photo/video modes deprecated
    lastChatAudio: null,
    captureIntervalMs: 200,
    lastServerError: null,
    duplexActive: false,
    mediaStream: null,
    streamError: null,
  });
  const streamingIdRef = useRef<string | null>(null);
  const duplexStreamingIdRef = useRef<string | null>(null);
  const duplexAudioQueueRef = useRef<string[]>([]);
  const configRef = useRef(state.config);
  configRef.current = state.config;
  const [mobileSettingsOpen, setMobileSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const playNextDuplexAudio = useCallback(() => {
    const q = duplexAudioQueueRef.current;
    if (q.length === 0) return;
    const b64 = q.shift();
    if (!b64) {
      playNextDuplexAudioRef.current();
      return;
    }
    try {
      const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const blob = new Blob([bin], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.onended = () => {
        URL.revokeObjectURL(url);
        playNextDuplexAudioRef.current();
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        playNextDuplexAudioRef.current();
      };
      audio.play().catch(() => {
        URL.revokeObjectURL(url);
        playNextDuplexAudioRef.current();
      });
    } catch {
      playNextDuplexAudioRef.current();
    }
  }, []);
  const playNextDuplexAudioRef = useRef(playNextDuplexAudio);
  playNextDuplexAudioRef.current = playNextDuplexAudio;

  const onEvent = useCallback((ev: ServerEvent) => {
    switch (ev.event) {
      case "session_ready":
        setState((s) => ({
          ...s,
          config: ev.data.config,
          messages: [
            { id: nextId(), role: "system", content: "Session ready. Upload media and start chatting." },
          ],
        }));
        break;
      case "upload_started":
        setState((s) => ({
          ...s,
          uploadStatus: "uploading",
          uploadMessage:
            ev.data.type === "image"
              ? `Uploading ${ev.data.count ?? ""} image(s)…`
              : "Uploading video…",
        }));
        break;
      case "upload_done":
        setState((s) => ({
          ...s,
          uploadStatus: "done",
          uploadMessage:
            ev.data.type === "image"
              ? `${ev.data.count} image(s) uploaded.`
              : `Video uploaded: ${ev.data.duration}s, ${ev.data.frame_count} frames.`,
        }));
        break;
      case "chat_started": {
        const id = nextId();
        streamingIdRef.current = id;
        setState((s) => ({
          ...s,
          isStreaming: true,
          messages: [...s.messages, { id, role: "assistant", content: "", streaming: true }],
        }));
        break;
      }
      case "chat_token":
        setState((s) => {
          const id = streamingIdRef.current;
          if (!id) return s;
          const idx = s.messages.findIndex((m) => m.id === id);
          if (idx === -1) return s;
          const next = [...s.messages];
          next[idx] = { ...next[idx], content: next[idx].content + ev.data.text };
          return { ...s, messages: next };
        });
        break;
      case "chat_done": {
        const id = streamingIdRef.current;
        streamingIdRef.current = null;
        setState((s) => ({
          ...s,
          messages: s.messages.map((m) =>
            m.id === id ? { ...m, content: ev.data.text, streaming: false } : m
          ),
          isStreaming: false,
        }));
        break;
      }
      case "chat_audio":
        setState((s) => ({
          ...s,
          lastChatAudio: { audio_b64: ev.data.audio_b64, sample_rate: ev.data.sample_rate },
        }));
        break;
      case "duplex_started":
        setState((s) => ({ ...s, duplexActive: true }));
        break;
      case "duplex_stopped":
        setState((s) => ({ ...s, duplexActive: false }));
        duplexStreamingIdRef.current = null;
        break;
      case "duplex_result": {
        const { text, audio_b64: audioB64, end_of_turn: endOfTurn } = ev.data;
        if (text) {
          setState((s) => {
            let id = duplexStreamingIdRef.current;
            const idx = id ? s.messages.findIndex((m) => m.id === id) : -1;
            if (idx >= 0) {
              const next = [...s.messages];
              next[idx] = {
                ...next[idx],
                content: next[idx].content + text,
                streaming: !endOfTurn,
              };
              if (endOfTurn) duplexStreamingIdRef.current = null;
              return { ...s, messages: next };
            }
            if (endOfTurn) {
              return {
                ...s,
                messages: [...s.messages, { id: nextId(), role: "assistant", content: text, streaming: false }],
              };
            }
            id = nextId();
            duplexStreamingIdRef.current = id;
            return {
              ...s,
              messages: [...s.messages, { id, role: "assistant", content: text, streaming: true }],
            };
          });
        } else if (endOfTurn) {
          duplexStreamingIdRef.current = null;
        }
        if (audioB64) {
          duplexAudioQueueRef.current.push(audioB64);
          playNextDuplexAudioRef.current();
        }
        break;
      }
      case "error":
        setState((s) => ({
          ...s,
          messages: s.messages.map((m) => (m.streaming ? { ...m, streaming: false } : m)),
          isStreaming: false,
          lastServerError: ev.data.message,
          ...(s.duplexActive ? { duplexActive: false } : {}),
        }));
        streamingIdRef.current = null;
        duplexStreamingIdRef.current = null;
        break;
      case "cleared":
        setState((s) => ({
          ...s,
          messages: [{ id: nextId(), role: "system", content: "Chat cleared." }],
        }));
        break;
      case "config_updated":
        setState((s) => ({ ...s, config: ev.data.config }));
        break;
    }
  }, []);

  const { status: wsStatus, lastError, send } = useWebSocket(onEvent);

  const setConfig = useCallback(
    (patch: Partial<ModelConfig>) => {
      setState((s) => ({ ...s, config: { ...s.config, ...patch } }));
      send({ action: "set_config", config: patch });
    },
    [send]
  );

  const setCurrentTab = useCallback((tab: Mode) => {
    setState((s) => ({ ...s, currentTab: tab }));
  }, []);

  const addSystemMessage = useCallback((text: string) => {
    setState((s) => ({
      ...s,
      messages: [...s.messages, { id: nextId(), role: "system", content: text }],
    }));
  }, []);

  const addUserMessage = useCallback((content: string) => {
    setState((s) => ({
      ...s,
      messages: [...s.messages, { id: nextId(), role: "user", content }],
    }));
  }, []);

  const clearLastChatAudio = useCallback(() => {
    setState((s) => ({ ...s, lastChatAudio: null }));
  }, []);

  const clearLastServerError = useCallback(() => {
    setState((s) => ({ ...s, lastServerError: null }));
  }, []);

  const setCaptureIntervalMs = useCallback((ms: number) => {
    setState((s) => ({ ...s, captureIntervalMs: ms }));
  }, []);

  const startDuplex = useCallback(async () => {
    try {
      setState((s) => ({ ...s, streamError: null }));
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      setState((s) => ({ ...s, mediaStream: stream, streamError: null }));
      const systemPrompt = configRef.current.system_prompt?.trim() || undefined;
      send({ action: "duplex_start", system_prompt: systemPrompt });
    } catch (err) {
      setState((s) => ({
        ...s,
        streamError: err instanceof Error ? err.message : String(err),
      }));
    }
  }, [send]);

  const stopDuplex = useCallback(() => {
    setState((s) => {
      const stream = s.mediaStream;
      if (stream) {
        send({ action: "duplex_stop" });
        stream.getTracks().forEach((t) => t.stop());
        return { ...s, mediaStream: null, streamError: null };
      }
      return s;
    });
  }, [send]);

  const value = useMemo<AppContextValue>(
    () => ({
      ...state,
      wsStatus: wsStatus as AppContextValue["wsStatus"],
      lastError,
      mobileSettingsOpen,
      setMobileSettingsOpen,
      sidebarOpen,
      setSidebarOpen,
      send,
      setConfig,
      setCurrentTab,
      addSystemMessage,
      addUserMessage,
      clearLastChatAudio,
      clearLastServerError,
      setCaptureIntervalMs,
      startDuplex,
      stopDuplex,
    }),
    [
      state,
      wsStatus,
      lastError,
      mobileSettingsOpen,
      sidebarOpen,
      send,
      setConfig,
      setCurrentTab,
      addSystemMessage,
      addUserMessage,
      clearLastChatAudio,
      clearLastServerError,
      setCaptureIntervalMs,
      startDuplex,
      stopDuplex,
    ]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
