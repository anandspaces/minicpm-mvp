import { useCallback, useEffect, useRef, useState } from "react";
import { useApp } from "@/context/AppContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Mic, Send, RotateCcw, X } from "lucide-react";

interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((e: { resultIndex: number; results: Iterable<{ isFinal: boolean; 0: { transcript: string } }> }) => void) | null;
  onend: (() => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  start: () => void;
  stop: () => void;
}

const SpeechRecognitionAPI =
  typeof window !== "undefined"
    ? ((window as unknown as { SpeechRecognition?: new () => SpeechRecognitionLike }).SpeechRecognition ||
        (window as unknown as { webkitSpeechRecognition?: new () => SpeechRecognitionLike }).webkitSpeechRecognition)
    : undefined;

export function ChatInput() {
  const {
    send,
    isStreaming,
    currentTab,
    messages,
    addUserMessage,
    lastChatAudio,
    config,
    clearLastChatAudio,
  } = useApp();
  const [input, setInput] = useState("");
  const [sttHint, setSttHint] = useState(
    SpeechRecognitionAPI ? "Focus to use voice input" : "Voice input not supported"
  );
  const [isListening, setIsListening] = useState(false);
  const voiceInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const sttFinalRef = useRef("");
  const isListeningRef = useRef(false);

  const lastUserContent = messages.filter((m) => m.role === "user").pop()?.content;

  const sendChat = useCallback(() => {
    if (isStreaming) return;
    const text = input.trim();
    if (!text) return;
    if (lastUserContent !== text) addUserMessage(text);
    send({ action: "chat", text, mode: currentTab });
    setInput("");
  }, [isStreaming, input, currentTab, lastUserContent, send, addUserMessage]);

  const startStt = useCallback(() => {
    if (!SpeechRecognitionAPI || isStreaming || isListening) return;
    sttFinalRef.current = "";
    setInput("");
    const rec = new SpeechRecognitionAPI();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = navigator.language || "en-US";
    rec.onresult = (e) => {
      let interim = "";
      const results = Array.from(e.results);
      for (let i = e.resultIndex; i < results.length; i++) {
        const transcript = results[i][0].transcript;
        if (results[i].isFinal) {
          sttFinalRef.current += transcript + " ";
          interim = "";
        } else {
          interim = transcript;
        }
      }
      setInput((sttFinalRef.current + interim).trim());
    };
    rec.onend = () => {
      if (isListeningRef.current && recognitionRef.current) recognitionRef.current.start();
    };
    rec.onerror = (e) => {
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        setSttHint("Microphone access denied");
        setIsListening(false);
      } else if (e.error !== "aborted") {
        setSttHint(`Voice error: ${e.error}`);
      }
    };
    try {
      rec.start();
      recognitionRef.current = rec;
      isListeningRef.current = true;
      setIsListening(true);
      setSttHint("Listening…");
    } catch (err) {
      setSttHint(`Could not start voice: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [isStreaming, isListening]);

  const stopStt = useCallback(() => {
    const rec = recognitionRef.current;
    isListeningRef.current = false;
    if (!rec) return;
    setIsListening(false);
    try {
      rec.stop();
    } catch {
      /**/
    }
    recognitionRef.current = null;
    setSttHint("Focus to use voice input");
  }, []);

  useEffect(() => {
    if (!isListening) return;
    return () => {
      recognitionRef.current?.stop();
      recognitionRef.current = null;
    };
  }, [isListening]);

  const prevAudioRef = useRef<string | null>(null);
  useEffect(() => {
    if (
      !lastChatAudio ||
      !config.use_tts ||
      lastChatAudio.audio_b64 === prevAudioRef.current
    )
      return;
    prevAudioRef.current = lastChatAudio.audio_b64;
    const bin = Uint8Array.from(atob(lastChatAudio.audio_b64), (c) => c.charCodeAt(0));
    const blob = new Blob([bin], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => {
      URL.revokeObjectURL(url);
      clearLastChatAudio();
    };
    audio.play().catch(() => clearLastChatAudio());
  }, [lastChatAudio, config.use_tts, clearLastChatAudio]);

  return (
    <div
      className={`flex shrink-0 gap-2 border-t border-border bg-card px-5 py-3 ${isListening ? "ring-1 ring-primary/50" : ""}`}
    >
      <div className="flex flex-1 flex-col gap-0.5">
        <Input
          className="min-h-0"
          placeholder="Type or speak…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendChat();
            }
          }}
          onFocus={() => {
            if (!isStreaming && SpeechRecognitionAPI) startStt();
          }}
          onBlur={() => {
            if (isListening) stopStt();
          }}
          disabled={isStreaming}
          aria-label="Chat message"
        />
        <span className="text-[11px] text-muted-foreground">{sttHint}</span>
      </div>
      <input
        ref={voiceInputRef}
        type="file"
        accept="audio/*,.wav"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (!file || isStreaming) return;
          addUserMessage("[Voice message]");
          const reader = new FileReader();
          reader.onload = () => {
            const dataUrl = reader.result as string;
            const b64 = dataUrl.includes(",") ? dataUrl.split(",")[1] : dataUrl;
            if (b64) send({ action: "chat_with_audio", audio_b64: b64, mode: currentTab });
          };
          reader.readAsDataURL(file);
        }}
      />
      <Button
        type="button"
        variant="secondary"
        size="icon"
        onClick={() => voiceInputRef.current?.click()}
        disabled={isStreaming}
        aria-label="Upload voice message"
      >
        <Mic className="size-4" />
      </Button>
      <Button type="button" onClick={sendChat} disabled={isStreaming} aria-label="Send message">
        <Send className="size-4" />
      </Button>
      <Button
        type="button"
        variant="secondary"
        size="icon"
        onClick={() => send({ action: "regenerate" })}
        disabled={isStreaming || messages.length < 2}
        aria-label="Regenerate last response"
      >
        <RotateCcw className="size-4" />
      </Button>
      <Button
        type="button"
        variant="secondary"
        size="icon"
        onClick={() => send({ action: "clear" })}
        disabled={isStreaming}
        aria-label="Clear chat"
      >
        <X className="size-4" />
      </Button>
    </div>
  );
}
