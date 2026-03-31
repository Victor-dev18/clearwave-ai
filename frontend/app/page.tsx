"use client";

import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [cleanAudioUrl, setCleanAudioUrl] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setCleanAudioUrl(null); // Reset the player if a new file is uploaded
    }
  };

  const handleCleanAudio = async () => {
    if (!file) return;

    setIsProcessing(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Send the file to our Python FastAPI server!
      const response = await fetch("http://127.0.0.1:8000/api/clean-audio", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Server error");

      // Receive the cleaned audio back
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setCleanAudioUrl(url);
    } catch (error) {
      console.error("Error cleaning audio:", error);
      alert("Failed to process audio. Make sure your Python backend is running!");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center p-6 font-sans">
      <div className="max-w-xl w-full bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl p-8">
        <div className="flex items-center justify-center gap-3 mb-2">
  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500">
    <path d="M2 12h4l3-9 5 18 3-9h5" />
  </svg>
  <h1 className="text-3xl font-bold text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
    ClearWave AI
  </h1>
</div>
        <p className="text-gray-400 text-center mb-8">Upload a noisy .wav file and let Deep Learning do the rest.</p>

        <div className="flex flex-col gap-6">
          {/* File Upload Area */}
          <div className="border-2 border-dashed border-gray-700 rounded-xl p-6 text-center hover:border-blue-500 transition-colors">
            <input 
              type="file" 
              accept="audio/wav" 
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-400
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-600 file:text-white
                hover:file:bg-blue-700 cursor-pointer"
            />
          </div>

          {/* Action Button */}
          <button 
            onClick={handleCleanAudio}
            disabled={!file || isProcessing}
            className={`w-full py-3 rounded-xl font-bold text-lg transition-all ${
              !file ? "bg-gray-800 text-gray-500 cursor-not-allowed" 
              : isProcessing ? "bg-blue-800 text-blue-200 animate-pulse" 
              : "bg-blue-600 hover:bg-blue-500 text-white shadow-[0_0_15px_rgba(37,99,235,0.5)]"
            }`}
          >
            {isProcessing ? "✨ AI is processing..." : "✨ Clean Audio"}
          </button>

          {/* Results & Download Area */}
          {cleanAudioUrl && (
            <div className="mt-4 p-6 bg-gray-800 rounded-xl border border-green-500/30 flex flex-col items-center gap-4">
              <h3 className="text-green-400 font-semibold">✅ Processing Complete!</h3>
              <audio controls src={cleanAudioUrl} className="w-full" />
              
              {/* THE DOWNLOAD BUTTON */}
              <a 
                href={cleanAudioUrl} 
                download={`Cleaned_${file?.name || 'audio.wav'}`}
                className="w-full text-center bg-gray-700 hover:bg-gray-600 py-2 rounded-lg font-medium transition-colors border border-gray-600"
              >
                ⬇️ Download Cleaned Audio
              </a>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}