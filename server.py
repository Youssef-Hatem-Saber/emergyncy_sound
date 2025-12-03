import uvicorn
import numpy as np
import librosa
import tensorflow as tf
import scipy.io.wavfile
import time
import os
from fastapi import FastAPI, Request

app = FastAPI()

# تحميل الموديل
# تأكد أن ملف emergency_sound_classifier.tflite موجود في نفس المجلد
INTERPRETER = tf.lite.Interpreter(model_path="emergency_sound_classifier.tflite")
INTERPRETER.allocate_tensors()

input_details = INTERPRETER.get_input_details()
output_details = INTERPRETER.get_output_details()

CLASSES = ["ambulance", "firetruck", "traffic"]

# إنشاء مجلد لحفظ التسجيلات القادمة (للتشخيص)
if not os.path.exists("received_audio"):
    os.makedirs("received_audio")

def pcm16_to_float32(raw_bytes):
    try:
        # 1. تحويل البايتات الخام إلى مصفوفة أرقام
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        
        if len(audio) == 0:
            return None

        # --- تعديل 1: إزالة الانحياز المستمر (DC Offset) ---
        # هذا يحل مشكلة أن الموديل يظن الصمت ضجيجاً عالياً
        audio = audio - np.mean(audio)

        # --- تعديل 2: توحيد مستوى الصوت (Normalization) ---
        # يجعل أعلى صوت في المقطع = 1.0 وأقل صوت = -1.0
        # هذا يساعد الموديل إذا كان الميكروفون صوته منخفضاً
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    except Exception as e:
        print(f"Error in pcm16_to_float32: {e}")
        return None


def extract_mfcc(audio, sr=16000, n_mfcc=40, max_len=40):
    try:
        # استخراج الخصائص
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)

        # ضبط الطول (Padding/Trimming)
        if len(mfcc) < max_len:
            mfcc = np.pad(mfcc, (0, max_len - len(mfcc)))
        else:
            mfcc = mfcc[:max_len]

        return mfcc
    except Exception as e:
        print(f"Error in extract_mfcc: {e}")
        return None


def predict(mfcc):
    mfcc = np.array(mfcc, dtype=np.float32).reshape(1, -1)
    INTERPRETER.set_tensor(input_details[0]["index"], mfcc)
    INTERPRETER.invoke()
    output = INTERPRETER.get_tensor(output_details[0]["index"])
    
    # الحصول على أعلى احتمال
    idx = int(np.argmax(output))
    confidence = float(output[0][idx])
    
    return CLASSES[idx], confidence


@app.post("/predict")
async def predict_raw(request: Request):
    try:
        # قراءة البيانات الخام من الطلب
        raw_bytes = await request.body()
        
        # --- خطوة التشخيص: حفظ الملف لنسمعه ---
        # سيتم حفظ الملف في مجلد received_audio
        # اسمع هذا الملف لتعرف هل الصوت واضح أم مجرد شوشرة
        timestamp = int(time.time())
        filename = f"received_audio/rec_{timestamp}.wav"
        
        # تحويل مؤقت للحفظ بصيغة WAV
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        scipy.io.wavfile.write(filename, 16000, audio_int16)
        print(f"--> Received audio saved to: {filename}")
        # --------------------------------------

        # المعالجة للتوقع
        audio = pcm16_to_float32(raw_bytes)
        if audio is None or len(audio) == 0:
            return {"error": "Invalid RAW PCM data"}

        mfcc = extract_mfcc(audio)
        if mfcc is None:
            return {"error": "MFCC error"}

        label, confidence = predict(mfcc)
        
        print(f"Prediction: {label} ({confidence:.2f})")
        return {"prediction": label, "confidence": confidence}

    except Exception as e:
        print(f"Server Error: {e}")
        return {"error": str(e)}


@app.get("/")
def root():
    return {"status": "RAW PCM server running with Signal Processing Fixes"}


if __name__ == "__main__":
    # تشغيل السيرفر
    uvicorn.run("server_main:app", host="0.0.0.0", port=8080, reload=True)
