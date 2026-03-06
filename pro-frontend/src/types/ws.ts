/**
 * WebSocket event and action types for MiniCPM-o 4.5 backend (pro/main.py).
 */

export type SessionReadyData = {
  max_frames: number;
  config: ModelConfig;
  streaming: boolean;
};

export type UploadStartedData = {
  type: "image" | "video";
  count?: number;
};

export type UploadDoneData =
  | { type: "image"; count: number }
  | { type: "video"; duration: number; frame_count: number; packing: number };

export type ChatStartedData = { frame_count: number; mode: Mode };

export type ChatTokenData = { text: string };
export type ChatDoneData = { text: string };

export type ChatAudioData = { audio_b64: string; sample_rate: number };

export type ErrorData = { message: string };

export type ClearedData = Record<string, never>;

export type ConfigUpdatedData = { config: ModelConfig };

export type DuplexStartedData = Record<string, never>;
export type DuplexStoppedData = Record<string, never>;
export type DuplexResultData = {
  is_listen: boolean;
  text: string;
  audio_b64: string | null;
  end_of_turn: boolean;
};

export type ModelConfig = {
  enable_thinking: boolean;
  do_sample: boolean;
  temperature: number;
  top_p: number;
  top_k: number;
  repetition_penalty: number;
  max_new_tokens: number;
  system_prompt: string;
  max_frames: number;
  video_fps: number;
  use_tts: boolean;
};

export type Mode = "photo" | "video" | "live";

export type ServerEvent =
  | { event: "session_ready"; data: SessionReadyData }
  | { event: "upload_started"; data: UploadStartedData }
  | { event: "upload_done"; data: UploadDoneData }
  | { event: "chat_started"; data: ChatStartedData }
  | { event: "chat_token"; data: ChatTokenData }
  | { event: "chat_done"; data: ChatDoneData }
  | { event: "chat_audio"; data: ChatAudioData }
  | { event: "duplex_started"; data: DuplexStartedData }
  | { event: "duplex_stopped"; data: DuplexStoppedData }
  | { event: "duplex_result"; data: DuplexResultData }
  | { event: "error"; data: ErrorData }
  | { event: "cleared"; data: ClearedData }
  | { event: "config_updated"; data: ConfigUpdatedData };

export type UploadImagesAction = { action: "upload_images"; images: string[] };
export type UploadVideoAction = { action: "upload_video"; data: string; fps?: number };
export type ChatWithAudioAction = { action: "chat_with_audio"; audio_b64: string; mode?: Mode };
export type FrameAction = { action: "frame"; data: string };
export type ChatAction = { action: "chat"; text: string; mode?: Mode };
export type RegenerateAction = { action: "regenerate" };
export type SetConfigAction = { action: "set_config"; config: Partial<ModelConfig> };
export type ClearAction = { action: "clear" };

export type DuplexStartAction = {
  action: "duplex_start";
  system_prompt?: string;
  ref_audio_b64?: string;
};
export type DuplexChunkAction = {
  action: "duplex_chunk";
  audio_b64: string;
  frame_b64?: string;
};
export type DuplexStopAction = { action: "duplex_stop" };

export type ClientAction =
  | UploadImagesAction
  | UploadVideoAction
  | ChatWithAudioAction
  | FrameAction
  | ChatAction
  | RegenerateAction
  | SetConfigAction
  | ClearAction
  | DuplexStartAction
  | DuplexChunkAction
  | DuplexStopAction;

export type MessageRole = "user" | "assistant" | "system";

export type ChatMessage = {
  id: string;
  role: MessageRole;
  content: string;
  streaming?: boolean;
};
