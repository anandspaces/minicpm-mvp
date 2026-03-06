import { useEffect } from "react";
import { Toaster } from "@/components/ui/sonner";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AppProvider, useApp } from "@/context/AppContext";
import { TopBar } from "@/components/Layout/TopBar";
import { Sidebar } from "@/components/Layout/Sidebar";
import { PhotoDropzone } from "@/components/Media/PhotoDropzone";
import { VideoUpload } from "@/components/Media/VideoUpload";
import { LiveWebcam } from "@/components/Media/LiveWebcam";
import { MessageList } from "@/components/Chat/MessageList";
import { ChatInput } from "@/components/Chat/ChatInput";
import { toast } from "sonner";

function AppContent() {
  const { lastServerError, clearLastServerError, currentTab, setCurrentTab } = useApp();

  useEffect(() => {
    if (lastServerError) {
      toast.error(lastServerError);
      clearLastServerError();
    }
  }, [lastServerError, clearLastServerError]);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background text-foreground">
      <TopBar />
      <Tabs
        value={currentTab}
        onValueChange={(v) => setCurrentTab(v as "photo" | "video" | "live")}
        className="flex flex-1 flex-col overflow-hidden"
      >
        <TabsList className="flex min-w-0 shrink-0 flex-nowrap justify-start gap-0 overflow-x-auto rounded-none border-b border-border bg-card px-3 sm:px-5">
          <TabsTrigger
            value="photo"
            className="shrink-0 rounded-none border-b-2 border-transparent text-xs sm:text-sm data-[state=active]:border-primary data-[state=active]:text-primary"
          >
            Photo Chat
          </TabsTrigger>
          <TabsTrigger
            value="video"
            className="shrink-0 rounded-none border-b-2 border-transparent text-xs sm:text-sm data-[state=active]:border-primary data-[state=active]:text-primary"
          >
            Video Chat
          </TabsTrigger>
          <TabsTrigger
            value="live"
            className="shrink-0 rounded-none border-b-2 border-transparent text-xs sm:text-sm data-[state=active]:border-primary data-[state=active]:text-primary"
          >
            Live Video
          </TabsTrigger>
        </TabsList>
        <div className="flex min-h-0 flex-1">
          <Sidebar />
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="shrink-0 border-b border-border px-3 py-4 sm:px-5">
              {currentTab === "photo" && <PhotoDropzone />}
              {currentTab === "video" && <VideoUpload />}
              {currentTab === "live" && <LiveWebcam />}
            </div>
            <div className="flex min-h-0 flex-1 flex-col">
              <MessageList />
              <ChatInput />
            </div>
          </div>
        </div>
      </Tabs>
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
