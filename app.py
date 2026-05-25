import numpy as np
import librosa
import io
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import soundfile as sf


# web上で呼び出すための設定
app = FastAPI()
@app.get("/")
def health():
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://acrolite.github.io",          # 以前のURL（バックアップ）
        "https://www.bpmestimator.com",        # 新ドメイン（wwwあり）
        "https://bpmestimator.com",            # 新ドメイン（wwwなし）
    ],
    allow_methods=["*"],
    allow_headers=["*"]
)

def bpm_estimate(
    y: np.ndarray,
    sr: int,
    hop_length: int = 128,
)-> dict:
    max_val = np.max(np.abs(y))
    print(f"DEBUG: 実際に受け取った最大振幅: {max_val}")
    if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
        print("DEBUG: 入力音声が空です。")
        return {"bpm_corrected": 0}

    _, y_perc = librosa.effects.hpss(y, margin=3.0)
    y_kick = librosa.effects.preemphasis(y_perc)
    onset_env = librosa.onset.onset_strength(y=y_kick, sr=sr,hop_length=hop_length, fmax=5000)
    
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length, ac_size=15.0, start_bpm=120, max_temp=240, min_tempo=50)
    bpm = float(tempo.item())
    print(f"DEBUG: librosa detect bpm: {bpm}")

    
    print(f"DEBUG: final_bpm: {bpm}")
    return{
        "bpm_corrected": float(bpm)
    }



    
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    print(f"Debug: Backend received: {len(data)}bytes")
    try:
        data_stream = io.BytesIO(data)
        y, native_sr = sf.read(data_stream)
        print(f"DEBUG: sf.read 成功! 形状={y.shape}, 元のSR={native_sr}")
        

    except Exception as e:
        print(f"DEBUG: sf.read 失敗: {str(e)}")

        try:
            print("DEBUG: librosaでの強制でコードを試みます")
            y, native_sr = librosa.load(io.BytesIO(data), sr=22050)
            print(f"DEBUG: librosa成功 長さ={len(y)}")
        except Exception as e2:
            print(f"DEBUG: すべてのでコードに失敗: {str(e2)}")
            return {"bpm_corrected":0, "error": "decode_failure"}
    if len(y) == 0:
        return {"bpm_corrected":0, "error": "decoded_array_is_empty"}

    result = bpm_estimate(y, native_sr, hop_length=128)
    return {"bpm_corrected": int(result["bpm_corrected"])}

