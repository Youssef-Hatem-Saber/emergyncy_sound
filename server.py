import uvicorn
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

# -------------------------
# Load TFLite Model
# -------------------------

INTERPRETER = tf.lite.Interpreter(model_path="emergency_sound_classifier.tflite")
INTERPRETER.allocate_tensors()

input_details = INTERPRETER.get_input_details()
output_details = INTERPRETER.get_output_details()

CLASSES = ["ambulance", "firetruck", "traffic"]


# -------------------------
# Extract MFCC (same as training)
# -------------------------
def extract_mfcc(audio_bytes, n_mfcc=40, max_len=40):
    try:
        # Load raw audio
        audio, sr = librosa.load(audio_bytes, sr=16000)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)

        # Pad / trim
        if len(mfcc) < max_len:
            mfcc = np.pad(mfcc, (0, max_len - len(mfcc)))
        else:
            mfcc = mfcc[:max_len]

        return mfcc

    except Exception as e:
        print("MFCC Error:", e)
        return None


# -------------------------
# Prediction Function
# -------------------------
def predict(mfcc):
    mfcc = np.array(mfcc, dtype=np.float32).reshape(1, -1)

    INTERPRETER.set_tensor(input_details[0]["index"], mfcc)
    INTERPRETER.invoke()

    output = INTERPRETER.get_tensor(output_details[0]["index"])
    predicted_idx = np.argmax(output)

    return CLASSES[predicted_idx], float(output[0][predicted_idx])


# -------------------------
# API Route
# -------------------------
@app.post("/predict")
async def predict_sound(file: UploadFile = File(...)):
    print("Received:", file.filename)

    # Read uploaded file
    audio_bytes = await file.read()

    # Save temporarily for librosa
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    mfcc = extract_mfcc(temp_path)

    if mfcc is None:
        return {"error": "Invalid audio file"}

    label, confidence = predict(mfcc)

    return {
        "prediction": label,
        "confidence": confidence
    }


@app.get("/")
def root():
    return {"status": "API is running!"}


# Local test only
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
