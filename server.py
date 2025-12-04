import uvicorn
import numpy as np
import librosa
import tensorflow as tf
import scipy.io.wavfile
import time
import os
from fastapi import FastAPI, Request

app = FastAPI()

# ---------------------------------------------------------
# إعدادات الموديل والفلاتر
# ---------------------------------------------------------
MODEL_PATH = "emergency_sound_classifier.tflite"
CLASSES = ["ambulance", "firetruck", "traffic"]

# 1. حد ارتفاع الصوت (Noise Gate)
NOISE_THRESHOLD = 0.1 

# 2. حد الشوشرة (ZCR Threshold) - الجديد!
# الوش العالي بيكون الـ ZCR بتاعه أعلى من 0.15 عادة
# الإسعاف بيكون أقل من 0.1
ZCR_THRESHOLD = 0.45

print("Loading TFLite Model...")
try:
    INTERPRETER = tf.lite.Interpreter(model_path=MODEL_PATH)
    INTERPRETER.allocate_tensors()
    input_details = INTERPRETER.get_input_details()
    output_details = INTERPRETER.get_output_details()
    print("Model Loaded Successfully.")
except Exception as e:
    print(f"ERROR: Could not load model form {MODEL_PATH}.")
    raise e

if not os.path.exists("received_audio"):
    os.makedirs("received_audio")


# ---------------------------------------------------------
# دوال المساعدة
# ---------------------------------------------------------

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
        print(f"Error in extract_mfcc: {e}")
        return None

def predict_from_mfcc(mfcc):
    mfcc_input = np.array(mfcc, dtype=np.float32).reshape(1, -1)
    INTERPRETER.set_tensor(input_details[0]["index"], mfcc_input)
    INTERPRETER.invoke()
    output = INTERPRETER.get_tensor(output_details[0]["index"])
    idx = int(np.argmax(output))
    confidence = float(output[0][idx])
    return CLASSES[idx], confidence

# ---------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------

@app.post("/predict")
async def predict_raw(request: Request):
    try:
        raw_bytes = await request.body()
        
        # 1. تحويل البيانات (float32)
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) == 0: return {"error": "Empty audio data"}

        # 2. إزالة DC Offset
        audio = audio - np.mean(audio)

        # 3. حساب الخصائص الفيزيائية للصوت
        max_amplitude = np.max(np.abs(audio))
        
        # حساب معدل تقاطع الصفر (ZCR) - كاشف الشوشرة
        # بنحسب متوسط عدد المرات اللي الموجة قطعت فيها خط الصفر
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))

        print(f"Stats -> Max Amp: {max_amplitude:.3f} | ZCR: {zcr:.3f}")

        # --- فلتر 1: الصوت الواطي (Silence Check) ---
        if max_amplitude < NOISE_THRESHOLD:
            print("--> Rejected: Silence/Low Volume")
            return {"prediction": "traffic", "confidence": 0.0, "note": "Low Volume"}

        if zcr > ZCR_THRESHOLD:
            print(f"--> Rejected: High Static Noise (ZCR={zcr:.3f})")
            return {"prediction": "traffic", "confidence": 0.0, "note": "Static Noise Detected"}

        # 4. تضخيم الصوت (فقط لو عدى الفلاتر)
        audio = audio / max_amplitude

        # حفظ الملف للمراجعة
        timestamp = int(time.time())
        filename = f"received_audio/rec_{timestamp}.wav"
        scipy.io.wavfile.write(filename, 16000, (audio * 32767).astype(np.int16))

        # 5. الذكاء الاصطناعي
        mfcc = extract_mfcc(audio)
        if mfcc is None: return {"error": "Failed to extract MFCC"}

        label, confidence = predict_from_mfcc(mfcc)
        
        print(f"Prediction: {label} ({confidence:.2f})")
        return {"prediction": label, "confidence": confidence}

    except Exception as e:
        print(f"Server Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("server_main:app", host="0.0.0.0", port=8080, reload=True)
