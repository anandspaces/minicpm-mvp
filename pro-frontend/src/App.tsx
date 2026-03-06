/**
 * Full duplex only — livestream mode. Photo and video modes are deprecated.
 */
import { useEffect } from "react";
import { Toaster } from "@/components/ui/sonner";
import { AppProvider, useApp } from "@/context/AppContext";
import { TopBar } from "@/components/Layout/TopBar";
import { Sidebar } from "@/components/Layout/Sidebar";
import { LiveWebcam } from "@/components/Media/LiveWebcam";
import { DuplexMicButton } from "@/components/Media/DuplexMicButton";
import { toast } from "sonner";
// deprecated: photo mode — PhotoDropzone
// deprecated: video mode — VideoUpload
// import { MessageList } from "@/components/Chat/MessageList";
// import { ChatInput } from "@/components/Chat/ChatInput";

function AppContent() {
  const { lastServerError, clearLastServerError } = useApp();

  useEffect(() => {
    if (lastServerError) {
      toast.error(lastServerError);
      clearLastServerError();
    }
  }, [lastServerError, clearLastServerError]);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background text-foreground">
      <TopBar />
      <div className="flex min-h-0 flex-1">
        <Sidebar />
        <div className="flex min-w-0 flex-1 flex-col">
          <LiveWebcam />
          <DuplexMicButton />
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppContent />
      <Toaster theme="dark" />
    </AppProvider>
  );
}
