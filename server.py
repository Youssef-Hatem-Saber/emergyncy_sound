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

MODEL_PATH = "emergency_sound_classifier.tflite"
CLASSES = ["ambulance", "firetruck", "traffic"]

# عتبة الضوضاء خفضناها جداً عشان تسمح بمرور الصوت القادم من ESP32
# القيمة 0.1 مناسبة لأن الـ ESP32 بيبعت حوالي 0.3
NOISE_THRESHOLD = 0.1 

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

# --- دوال المساعدة ---

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

# --- API Endpoint ---

@app.post("/predict")
async def predict_raw(request: Request):
    try:
        raw_bytes = await request.body()
        
        # 1. تحويل البيانات (float32)
        # القسمة على 32768 هي المعيار، والـ ESP32 ضرب في 5، فالنتيجة القصوى حوالي 0.3
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        if len(audio) == 0: return {"error": "Empty audio data"}

        # 2. إزالة DC Offset
        audio = audio - np.mean(audio)

        # 3. قياس قوة الصوت
        max_amplitude = np.max(np.abs(audio))
        print(f"Stats -> Max Amp: {max_amplitude:.3f}")

        # --- الحل الجذري لمشكلة الثبات ---
        # بدلاً من رفض الصوت الواطي، سنقوم بمعالجته بحذر
        
        if max_amplitude < NOISE_THRESHOLD:
            # السيناريو 1: الصوت واطي جداً (صمت أو وش)
            # لا نضخمه أبداً! نتركه ضعيفاً كما هو ونمرره للموديل
            # الموديل لما يشوف أرقام صغيرة جداً غالباً بيميل للـ Traffic
            print("--> Low Volume. Passing without normalization (Preventing Noise Amplification).")
            # لا يوجد كود audio = audio / max_amplitude هنا
        
        else:
            # السيناريو 2: الصوت عالي ومحترم (أكبر من 0.1)
            # هنا نضخمه ليوصل لـ 1.0 عشان الموديل يسمعه بوضوح
            print("--> Good Volume. Normalizing...")
            audio = audio / max_amplitude
        
        # ------------------------------------

        # حفظ الملف للمراجعة
        timestamp = int(time.time())
        filename = f"received_audio/rec_{timestamp}.wav"
        scipy.io.wavfile.write(filename, 16000, (audio * 32767).astype(np.int16))

        # 4. التوقع
        mfcc = extract_mfcc(audio)
        if mfcc is None: return {"error": "Failed to extract MFCC"}

        label, confidence = predict_from_mfcc(mfcc)
        
        print(f"Prediction: {label} ({confidence:.2f})")
        
        # إضافة تحذير في الرد لو الصوت كان واطي والموديل طلع نتيجة "اسعاف"
        note = "Normal"
        if max_amplitude < NOISE_THRESHOLD:
            note = "Low Volume Warning (Result might be inaccurate)"

        return {"prediction": label, "confidence": confidence, "note": note}

    except Exception as e:
        print(f"Server Error: {e}")
        return {"error": str(e)}

# دوال تحميل الملفات
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
