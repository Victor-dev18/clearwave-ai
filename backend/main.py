from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import librosa
import numpy as np
import soundfile as sf
import os
from model import UNet # This imports your model.py file!

app = FastAPI()

# This allows your future Next.js frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🧠 Loading AI Brain into the Server...")
model = UNet()
model.load_state_dict(torch.load("denoising_unet.pth", weights_only=True))
model.eval()
print("✅ Brain Loaded! Server is ready.")

# This is the "Endpoint" your frontend will send audio to
@app.post("/api/clean-audio")
async def clean_audio(file: UploadFile = File(...)):
    print(f"📥 Received file: {file.filename}")
    
    # 1. Save uploaded file temporarily
    input_path = f"temp_{file.filename}"
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    # 2. Process Audio
    audio_data, sr = librosa.load(input_path, sr=16000)
    stft_data = librosa.stft(audio_data, n_fft=512)
    mag, phase = librosa.magphase(stft_data)
    
    mag_tensor = torch.tensor(mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        clean_mag_tensor = model(mag_tensor)
        
    clean_mag = clean_mag_tensor.squeeze().numpy()
    reconstructed_stft = clean_mag * phase
    cleaned_audio = librosa.istft(reconstructed_stft)

    # 3. Save cleaned audio
    output_path = f"clean_{file.filename}"
    sf.write(output_path, cleaned_audio, sr)

    # 4. Cleanup temp input and return the clean file
    os.remove(input_path)
    print("📤 Sending clean file back to user!")
    return FileResponse(output_path, media_type="audio/wav", filename=output_path)