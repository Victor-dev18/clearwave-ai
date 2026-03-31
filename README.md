# 🎧 ClearWave AI

> **Studio-Quality Audio, Powered by Deep Learning.**

ClearWave AI is a full-stack, machine learning application designed to intelligently isolate human speech and remove complex background noise from audio files. Unlike traditional DSP (Digital Signal Processing) filters that blindly cut frequencies, ClearWave utilizes a custom-trained Convolutional Neural Network (U-Net) to map and eliminate noise profiles dynamically.

---

## 🚀 Features

* **Deep Learning Audio Processing:** Uses a custom PyTorch U-Net architecture trained on audio spectrograms.
* **Phase Reconstruction:** Implements STFT (Short-Time Fourier Transform) and iSTFT to preserve the original audio phase, preventing the "robotic" sound common in basic AI audio filters.
* **Full-Stack Architecture:** A decoupled Next.js frontend communicating with a high-performance Python/FastAPI backend.
* **Seamless UI/UX:** Dark-mode interface built with Tailwind CSS, featuring one-click processing and instant downloads.

---

## 🧠 How It Works (The AI Pipeline)

1. **Input:** The user uploads a noisy `.wav` file via the Next.js client.
2. **Feature Extraction:** The FastAPI backend uses `librosa` to compute the STFT, separating the audio into **Magnitude** and **Phase**.
3. **Inference:** The AI model (U-Net) only analyzes the Magnitude, identifying structural patterns of speech vs. the chaotic scatter of white noise.
4. **Reconstruction:** The cleaned Magnitude is multiplied back with the original Phase. An inverse STFT (iSTFT) is applied to generate the final, clean time-domain waveform.

---

## 🛠️ Tech Stack

### **Frontend (Web UI)**

* Next.js (React)
* Tailwind CSS
* TypeScript
* Deployed on Vercel

### **Backend (API & Inference)**

* FastAPI (Python)
* Uvicorn
* Deployed on Render

### **Machine Learning & DSP**

* PyTorch (Deep Learning Framework)
* Librosa (Audio & Signal Processing)
* NumPy

---

## 💻 Local Installation & Setup

If you want to run this project locally, you will need two terminal windows to run the frontend and backend simultaneously.

### 1. Clone the Repository

```bash
git clone https://github.com/Victor-dev18/clearwave-ai.git
cd clearwave-ai
```

### 2. Start the Backend (FastAPI)

Navigate to the backend folder and install the required Python libraries. (Note: You must have Python installed).

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will start running at [http://localhost:8000](http://localhost:8000)

### 3. Start the Frontend (Next.js)

Open a second terminal, navigate to the frontend folder, and install the Node dependencies.

```bash
cd frontend
npm install
npm run dev
```

The web interface will be available at [http://localhost:3000](http://localhost:3000)

---

## 📈 Future Improvements (Roadmap)

* [ ] Dataset Expansion: Train the U-Net on a massive dataset (e.g., Mozilla Common Voice) with diverse noise profiles (traffic, wind, typing).
* [ ] Advanced Metrics: Implement objective evaluation metrics like PESQ (Perceptual Evaluation of Speech Quality) and STOI (Short-Time Objective Intelligibility) in the training loop.
* [ ] Real-Time Streaming: Optimize the model architecture (e.g., using a smaller CNN or quantization) to process live microphone input with minimal latency.

---

## 👨‍💻 Author

Designed and developed by **Victor Devanand Kongala**.
