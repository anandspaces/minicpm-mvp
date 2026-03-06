import { useEffect, useRef } from "react";
import { useApp } from "@/context/AppContext";
import { MessageBubble } from "./MessageBubble";

export function MessageList() {
  const { messages } = useApp();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto overflow-x-hidden"
      role="log"
      aria-live="polite"
    >
      <div className="flex flex-col gap-3 px-5 py-4">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
      </div>
    </div>
  );
}
