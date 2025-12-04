import uvicorn
import numpy as np
import librosa
import tensorflow as tf
import scipy.io.wavfile
import time
import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

app = FastAPI()

# ---------------------------------------------------------
# إعدادات الموديل (تم تعديل الحساسية للأماكن المزعجة)
# ---------------------------------------------------------
MODEL_PATH = "emergency_sound_classifier.tflite"
CLASSES = ["ambulance", "firetruck", "traffic"]

# 1. رفعنا حد الضوضاء لـ 0.35 
# (أي صوت أقل من 0.35 سيعتبر ضوضاء خلفية ولن يتم تحليله)
# بما أن الضوضاء عندك 0.4، قد تحتاج لرفع هذا الرقم لـ 0.45 لو لسه بيلقط
NOISE_THRESHOLD = 0.35 

# تم إلغاء ZCR_THRESHOLD لأن المكان عندك فيه وش طبيعي عالي

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
# API Endpoints
# ---------------------------------------------------------

@app.post("/predict")
async def predict_raw(request: Request):
    try:
        raw_bytes = await request.body()
        
        # 1. تحويل البيانات (float32)
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) == 0: return {"error": "Empty audio data"}

        # 2. إزالة DC Offset (مهم جداً لتقليل القراءة الخاطئة للضوضاء)
        audio = audio - np.mean(audio)

        # 3. تحليل مستوى الصوت
        max_amplitude = np.max(np.abs(audio))
        
        print(f"Stats -> Max Amp: {max_amplitude:.3f}")

        # --- فلتر 1: تجاهل الضوضاء الخلفية العالية ---
        # إذا كان الصوت أقل من الحد المسموح (0.35)، نعتبره Traffic
        if max_amplitude < NOISE_THRESHOLD:
            print(f"--> Rejected: Below Threshold ({NOISE_THRESHOLD})")
            return {"prediction": "traffic", "confidence": 0.0, "note": "High Background Noise Ignored"}

        # 4. تضخيم ذكي (Smart Normalization)
        # بدلاً من تضخيم الصوت لأقصى حد (مما يشوه الإشارة)، نضخمه فقط إذا كان يحتاج ذلك
        # ونضع سقفاً للتضخيم (مثلاً لا نضرب في أكثر من 2x) لتجنب تفجير الضوضاء
        
        target_level = 0.8 # المستوى المستهدف
        gain = target_level / (max_amplitude + 0.0001) # حساب مقدار التضخيم المطلوب
        
        # لو الجين المطلوب كبير جداً (أكثر من 3 أضعاف)، نحدده بـ 3 فقط
        # عشان منضخمش الوش لمستويات مرعبة
        if gain > 3.0: 
            gain = 3.0
            
        audio = audio * gain
        
        # التأكد من عدم تجاوز الحدود -1 و 1 (Clipping Protection)
        audio = np.clip(audio, -1.0, 1.0)

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

@app.get("/files")
def list_files():
    files = []
    if os.path.exists("received_audio"):
        files = os.listdir("received_audio")
        files.sort(reverse=True)
    return {"files": files}

@app.get("/files/{filename}")
def get_file(filename: str):
    file_path = f"received_audio/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run("server_main:app", host="0.0.0.0", port=8080, reload=True)
