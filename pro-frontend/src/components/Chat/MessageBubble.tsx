import type { ChatMessage as Msg } from "@/types/ws";
import { cn } from "@/lib/utils";

export function MessageBubble({ message }: { message: Msg }) {
  const { role, content, streaming } = message;
  const isUser = role === "user";
  const isSystem = role === "system";

  if (isSystem) {
    return (
      <div className="flex justify-center py-1 text-xs text-muted-foreground" role="status">
        {content}
      </div>
    );
  }

  return (
    <div
      className={cn(
        "max-w-[85%] rounded-lg px-3.5 py-2.5 text-[13px] leading-relaxed whitespace-pre-wrap break-words",
        isUser
          ? "ml-auto bg-secondary text-secondary-foreground rounded-br-sm"
          : "rounded-bl-sm bg-muted",
        streaming && !isUser && "border-l-2 border-primary"
      )}
      data-role={role}
    >
      {content}
    </div>
  );
}
