import uvicorn
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
import soundfile as sf

app = FastAPI()

INTERPRETER = tf.lite.Interpreter(model_path="emergency_sound_classifier.tflite")
INTERPRETER.allocate_tensors()

input_details = INTERPRETER.get_input_details()
output_details = INTERPRETER.get_output_details()

CLASSES = ["ambulance", "firetruck", "traffic"]


def extract_from_wav(raw_bytes):
    try:
        audio, sr = sf.read(BytesIO(raw_bytes), dtype="float32")
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return audio
    except:
        return None


def extract_from_raw_pcm(raw_bytes):
    try:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0
        return audio
    except:
        return None


def extract_mfcc(audio, sr=16000, n_mfcc=40, max_len=40):
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)

        if len(mfcc) < max_len:
            mfcc = np.pad(mfcc, (0, max_len - len(mfcc)))
        else:
            mfcc = mfcc[:max_len]

        return mfcc
    except Exception as e:
        print("MFCC Error:", e)
        return None


def predict(mfcc):
    mfcc = np.array(mfcc, dtype=np.float32).reshape(1, -1)
    INTERPRETER.set_tensor(input_details[0]["index"], mfcc)
    INTERPRETER.invoke()
    output = INTERPRETER.get_tensor(output_details[0]["index"])
    idx = int(np.argmax(output))
    return CLASSES[idx], float(output[0][idx])


@app.post("/predict")
async def predict_sound(file: UploadFile = File(...)):
    raw_bytes = await file.read()

    audio = extract_from_wav(raw_bytes)
    if audio is None:
        audio = extract_from_raw_pcm(raw_bytes)

    if audio is None:
        return {"error": "Invalid audio format."}

    mfcc = extract_mfcc(audio)
    if mfcc is None:
        return {"error": "Failed to extract MFCC"}

    label, confidence = predict(mfcc)
    return {"prediction": label, "confidence": confidence}


@app.get("/")
def root():
    return {"status": "API running", "ok": True}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
