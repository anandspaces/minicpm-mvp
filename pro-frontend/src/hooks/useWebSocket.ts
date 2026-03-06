import { useCallback, useEffect, useRef, useState } from "react";
import type { ClientAction, ServerEvent } from "@/types/ws";

const RECONNECT_DELAY_MS = 2000;

function getWsUrl(): string {
  // Full WebSocket URL (e.g. ws://localhost:8084/ws)
  const wsUrl = import.meta.env.VITE_WS_URL;
  if (wsUrl) return wsUrl;
  // Base API URL (e.g. http://localhost:8084) -> derive ws URL
  const apiUrl = import.meta.env.VITE_API_URL;
  if (apiUrl) {
    const u = apiUrl.replace(/\/+$/, "");
    const wsProto = u.startsWith("https") ? "wss" : "ws";
    const host = u.replace(/^https?:\/\//, "");
    return `${wsProto}://${host}/ws`;
  }
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws`;
}

export type WsStatus = "connecting" | "open" | "closed" | "error";

export function useWebSocket(onEvent: (ev: ServerEvent) => void) {
  const [status, setStatus] = useState<WsStatus>("connecting");
  const [lastError, setLastError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const connect = useCallback(() => {
    const url = getWsUrl();
    setLastError(null);
    setStatus("connecting");
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setStatus("open");
      setLastError(null);
    };

    ws.onclose = () => {
      wsRef.current = null;
      setStatus("closed");
      reconnectTimerRef.current = setTimeout(() => {
        reconnectTimerRef.current = null;
        connect();
      }, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
      setStatus("error");
      setLastError("WebSocket error");
    };

    ws.onmessage = (e) => {
      try {
        const ev = JSON.parse(e.data) as ServerEvent;
        if (ev && typeof ev.event === "string") onEventRef.current(ev);
      } catch {
        setLastError("Invalid message");
      }
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setStatus("closed");
    };
  }, [connect]);

  const send = useCallback((obj: ClientAction) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }, []);

  return { status, lastError, send };
}
