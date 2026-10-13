
import whisper
import speech_recognition as sr
import tempfile
import os

# Whisper 모델 로드
model = whisper.load_model("base")
recognizer = sr.Recognizer()

def record_and_transcribe():
    try:
        with sr.Microphone() as source:
            print("주변 소음 측정 중...")
            recognizer.adjust_for_ambient_noise(source, duration=1)

            print("환자 증상 음성 입력 대기 중...")
            audio = recognizer.listen(
                source,
                timeout=10,
                phrase_time_limit=30
            )

        # 임시 WAV 파일 생성
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp:
            audio_path = temp.name
            temp.write(audio.get_wav_data())

        try:
            print("Whisper 음성 인식 중...")

            result = model.transcribe(
                audio_path,
                language="ko",
                fp16=False
            )

            return result["text"].strip()

        finally:
            os.remove(audio_path)

    except sr.WaitTimeoutError:
        print("음성이 감지되지 않았습니다.")
        return None

    except Exception as e:
        print(f"오류 발생: {e}")
        return None


patient_symptom = record_and_transcribe()

if patient_symptom:
    print(f"인식된 텍스트: {patient_symptom}")
