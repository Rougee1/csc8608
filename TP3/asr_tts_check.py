import torch
import torchaudio
from transformers import pipeline


def main():
    wav_path = "TP3/outputs/tts_reply_call_01.wav"
    model_id = "openai/whisper-small"

    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=device,
    )

    inp = {"array": wav.squeeze(0).numpy(), "sampling_rate": sr}
    result = asr(inp)
    text = result.get("text", "").strip()

    original = (
        "Thank you for contacting us. Your return request has been received "
        "and we will process it within two business days."
    )

    print("original_text:", original)
    print("asr_text:", text)
    print("match:", text.lower().strip() == original.lower().strip())


if __name__ == "__main__":
    main()
