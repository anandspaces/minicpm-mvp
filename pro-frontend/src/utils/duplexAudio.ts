/**
 * Capture ~1 s of microphone audio from a MediaStream, resample to 16 kHz mono,
 * encode as WAV, and return base64. Used for duplex_chunk payloads.
 */

const TARGET_SAMPLE_RATE = 16000;
const CHUNK_MS = 1000;

function audioBufferToWavBase64(buffer: AudioBuffer, sampleRate: number): string {
  const numChannels = 1;
  const format = 1; // PCM
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataLength = buffer.length * numChannels * bytesPerSample;
  const headerLength = 44;
  const totalLength = headerLength + dataLength;

  const arrayBuffer = new ArrayBuffer(totalLength);
  const view = new DataView(arrayBuffer);
  const offset = 0;

  function writeStr(pos: number, str: string) {
    for (let i = 0; i < str.length; i++) view.setUint8(pos + i, str.charCodeAt(i));
  }

  writeStr(offset, "RIFF");
  view.setUint32(offset + 4, totalLength - 8, true);
  writeStr(offset + 8, "WAVE");
  writeStr(offset + 12, "fmt ");
  view.setUint32(offset + 16, 16, true); // fmt chunk size
  view.setUint16(offset + 20, format, true); // PCM
  view.setUint16(offset + 22, numChannels, true);
  view.setUint32(offset + 24, sampleRate, true);
  view.setUint32(offset + 28, byteRate, true);
  view.setUint16(offset + 32, blockAlign, true);
  view.setUint16(offset + 34, bitsPerSample, true);
  writeStr(offset + 36, "data");
  view.setUint32(offset + 40, dataLength, true);

  const ch0 = buffer.getChannelData(0);
  const ch1 = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : null;
  for (let i = 0; i < buffer.length; i++) {
    let s = ch0[i];
    if (ch1) s = (s + ch1[i]) / 2;
    s = Math.max(-1, Math.min(1, s));
    const v = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(headerLength + i * 2, v, true);
  }

  const bytes = new Uint8Array(arrayBuffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

async function resampleTo16kMono(buffer: AudioBuffer): Promise<AudioBuffer> {
  if (buffer.sampleRate === TARGET_SAMPLE_RATE && buffer.numberOfChannels === 1) return buffer;
  const duration = buffer.duration * (TARGET_SAMPLE_RATE / buffer.sampleRate);
  const sampleCount = Math.round(duration * TARGET_SAMPLE_RATE);
  const ctx = new OfflineAudioContext(1, sampleCount, TARGET_SAMPLE_RATE);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start(0);
  return ctx.startRendering();
}

const getRecorderOptions = (): MediaRecorderOptions => {
  if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus"))
    return { mimeType: "audio/webm;codecs=opus", audioBitsPerSecond: 64000 };
  if (MediaRecorder.isTypeSupported("audio/webm")) return { mimeType: "audio/webm" };
  return {};
};

export async function captureAudioChunk(stream: MediaStream): Promise<string> {
  const ctx = new AudioContext();
  const recorder = new MediaRecorder(stream, getRecorderOptions());
  const chunks: Blob[] = [];

  return new Promise((resolve, reject) => {
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    recorder.onstop = async () => {
      try {
        const blob = new Blob(chunks, { type: recorder.mimeType });
        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
        const resampled = await resampleTo16kMono(audioBuffer);
        const b64 = audioBufferToWavBase64(resampled, TARGET_SAMPLE_RATE);
        await ctx.close();
        resolve(b64);
      } catch (err) {
        ctx.close().catch(() => {});
        reject(err);
      }
    };
    recorder.onerror = () => {
      ctx.close().catch(() => {});
      reject(new Error("MediaRecorder error"));
    };
    recorder.start(100);
    setTimeout(() => {
      if (recorder.state === "recording") recorder.stop();
    }, CHUNK_MS);
  });
}

export const DUPLEX_CHUNK_MS = CHUNK_MS;
