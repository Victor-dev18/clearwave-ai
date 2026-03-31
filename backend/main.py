from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import librosa
import numpy as np
import soundfile as sf
import os
import gc  # NEW: Python's Garbage Collector

from model import UNet

# NEW: Restrict PyTorch to a single thread to save massive amounts of RAM
torch.set_num_threads(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🧠 Loading AI Brain into the Server...")
model = UNet()
# Load model straight to CPU memory efficiently
model.load_state_dict(torch.load("denoising_unet.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()
print("✅ Brain Loaded! Server is ready.")

@app.post("/api/clean-audio")
async def clean_audio(file: UploadFile = File(...)):
    print(f"📥 Received file: {file.filename}")
    
    input_path = f"temp_{file.filename}"
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    # 1. Process Audio
    audio_data, sr = librosa.load(input_path, sr=16000)
    stft_data = librosa.stft(audio_data, n_fft=512)
    mag, phase = librosa.magphase(stft_data)
    
    mag_tensor = torch.tensor(mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # NEW: Delete variables we no longer need IMMEDIATELY
    del audio_data
    del stft_data
    gc.collect() # Force Python to clear RAM
    
    # 2. Run Inference
    with torch.no_grad():
        clean_mag_tensor = model(mag_tensor)
        
    clean_mag = clean_mag_tensor.squeeze().numpy()
    reconstructed_stft = clean_mag * phase
    cleaned_audio = librosa.istft(reconstructed_stft)

    # NEW: Delete massive AI tensors
    del mag_tensor
    del clean_mag_tensor
    del mag
    del phase
    gc.collect() # Force Python to clear RAM again

    # 3. Save cleaned audio
    output_path = f"clean_{file.filename}"
    sf.write(output_path, cleaned_audio, sr)

    os.remove(input_path)
    print("📤 Sending clean file back to user!")
    return FileResponse(output_path, media_type="audio/wav", filename=output_path)
