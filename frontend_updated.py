import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import uuid
import pygame
import websocket
import subprocess
import json
import io
import wave
import struct

# === HANDLE BUNDLED PATHS ===
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS  # PyInstaller temp directory
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_DIR = os.path.join(BASE_DIR, "audio_samples")
FFPROBE_PATH = os.path.join(BASE_DIR, "ffmpeg_bin", "ffprobe.exe")
WS_URL = "ws://15.157.158.90:8080/ws"

def get_audio_duration_ffmpeg(path):
    try:
        result = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return None

class AudioStreamerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Streamer")
        self.selected_file = None
        self.is_playing = False

        pygame.mixer.init()

        self.file_listbox = tk.Listbox(root, width=50, height=10)
        self.file_listbox.pack(pady=10)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_select)

        self.preview_button = ttk.Button(root, text="▶ Preview", command=self.preview_audio)
        self.preview_button.pack(pady=5)

        self.stream_button = ttk.Button(root, text="📤 Stream to Server", command=self.stream_audio_thread)
        self.stream_button.pack(pady=5)

        self.status_text = tk.Text(root, height=15, width=80, state='disabled')
        self.status_text.pack(pady=10)

        self.load_audio_files()

    def log(self, message):
        def append():
            self.status_text.config(state='normal')
            self.status_text.insert('end', f"{time.strftime('%H:%M:%S')} - {message}\n")
            self.status_text.yview('end')
            self.status_text.config(state='disabled')
        self.root.after(0, append)

    def load_audio_files(self):
        try:
            files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('_16k.wav')]
        except FileNotFoundError:
            self.log("❌ audio_samples folder not found.")
            return

        self.file_listbox.delete(0, tk.END)
        for f in files:
            self.file_listbox.insert(tk.END, f)

    def on_select(self, event):
        selected_index = self.file_listbox.curselection()
        if selected_index:
            self.selected_file = os.path.join(AUDIO_DIR, self.file_listbox.get(selected_index[0]))

    def preview_audio(self):
        if not self.selected_file:
            messagebox.showwarning("No file selected", "Please select an audio file.")
            return

        try:
            if self.is_playing:
                pygame.mixer.music.stop()
                self.is_playing = False
                self.preview_button.config(text="▶ Preview")
                self.log("⏹️ Playback stopped.")
            else:
                pygame.mixer.music.load(self.selected_file)
                pygame.mixer.music.play()
                self.is_playing = True
                self.preview_button.config(text="⏹ Stop")
                self.log(f"▶ Playing: {os.path.basename(self.selected_file)}")
                self.root.after(100, self.check_playback_status)
        except Exception as e:
            self.log(f"❌ Error playing file: {e}")

    def check_playback_status(self):
        if self.is_playing and not pygame.mixer.music.get_busy():
            self.is_playing = False
            self.preview_button.config(text="▶ Preview")
            self.log("✅ Playback finished.")
        elif self.is_playing:
            self.root.after(100, self.check_playback_status)

    def stream_audio_thread(self):
        threading.Thread(target=self.stream_audio, daemon=True).start()

    def get_pcm_from_wav(self, wav_path):
        """Extract raw PCM data from WAV file"""
        with wave.open(wav_path, 'rb') as wav_file:
            # Get audio parameters
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            self.log(f"📊 WAV info: {channels} channels, {sample_width} bytes/sample, {framerate} Hz, {n_frames} frames")
            
            # Read all frames (raw PCM data)
            pcm_data = wav_file.readframes(n_frames)
            return pcm_data

    def stream_audio(self):
        if not self.selected_file:
            self.log("⚠️ No file selected")
            return

        try:
            ws = websocket.WebSocket()
            self.log("🔌 Connecting to WebSocket...")
            ws.connect(WS_URL)
            self.log("✅ Connected")

            duration = get_audio_duration_ffmpeg(self.selected_file)
            if duration:
                self.log(f"📁 Selected audio length: {duration:.2f}s")

            # Get raw PCM data from WAV file
            pcm_data = self.get_pcm_from_wav(self.selected_file)
            
            chunk_size = 64 * 1024  # 64KB chunks as backend expects
            chunk_num = 0

            # Send PCM data in chunks
            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i:i + chunk_size]
                ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                chunk_num += 1
                self.log(f"🚀 Sent chunk {chunk_num} ({len(chunk)} bytes)")
                time.sleep(0.05)  # Small delay between chunks

            # Send end_stream message
            ws.send(json.dumps({"type": "end_stream"}))
            self.log("🏁 Sent end_stream control message")

            # Receive response
            audio_buffer = bytearray()
            start_time = time.time()
            self.log("⏳ Waiting for response...")

            received_metadata = False
            chunk_count = 0

            while True:
                # Increased timeout to 60 seconds for processing
                if time.time() - start_time > 60:
                    self.log("⚠️ Timeout waiting for audio response (60s)")
                    break

                try:
                    response = ws.recv()
                    
                    if isinstance(response, str):
                        self.log(f"📩 Received JSON: {response}")
                        try:
                            data = json.loads(response)
                            if data.get("type") == "metadata":
                                info = data.get("audio_info", {})
                                self.log(f"ℹ️ Audio format: {info.get('format')} at {info.get('sample_rate')} Hz")
                                received_metadata = True
                                # Reset timer after metadata since audio is coming
                                start_time = time.time()
                            elif data.get("type") == "audio_complete":
                                self.log("✅ Received audio_complete signal")
                                break
                            elif data.get("type") == "error":
                                self.log(f"❌ Server error: {data.get('message')}")
                                break
                        except json.JSONDecodeError:
                            self.log(f"⚠️ Invalid JSON: {response}")
                    
                    elif isinstance(response, bytes):
                        if received_metadata:
                            chunk_count += 1
                            # Only log every 10th chunk to reduce spam
                            if chunk_count % 10 == 1:
                                self.log(f"📥 Receiving audio chunks... ({chunk_count} received, {len(audio_buffer)} bytes total)")
                            audio_buffer.extend(response)
                        else:
                            self.log("⚠️ Received audio before metadata")
                    
                    else:
                        self.log(f"⚠️ Received unknown type: {type(response)}")
                        
                except websocket.WebSocketTimeoutException:
                    elapsed = time.time() - start_time
                    self.log(f"⏱️ Still waiting... ({elapsed:.1f}s elapsed)")
                    continue
                except websocket.WebSocketException as e:
                    self.log(f"❌ WebSocket error: {e}")
                    break

            ws.close()
            self.log("🔒 WebSocket closed.")

            if audio_buffer:
                self.log(f"🎧 Total audio received: {len(audio_buffer)} bytes ({len(audio_buffer)/1024/1024:.1f} MB)")
                
                # Save the complete WAV file
                tmp_path = os.path.join(AUDIO_DIR, f"tts_response_{uuid.uuid4().hex[:8]}.wav")
                with open(tmp_path, 'wb') as f:
                    f.write(audio_buffer)
                self.log(f"💾 Saved response to: {tmp_path}")

                # Play the audio
                try:
                    pygame.mixer.music.load(tmp_path)
                    pygame.mixer.music.play()
                    self.log("🔊 Playing response...")
                    
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    self.log("✅ Playback finished.")
                except Exception as e:
                    self.log(f"❌ Error playing audio: {e}")
            else:
                self.log("⚠️ No audio received from server")

        except websocket.WebSocketException as e:
            self.log(f"❌ WebSocket error: {e}")
        except Exception as e:
            self.log(f"❌ Error during streaming: {e}")

# === RUN THE APP ===
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioStreamerApp(root)
    root.mainloop()