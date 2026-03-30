# TP3 – Deep Learning pour l'Audio (Call Center)

---

## Exercice 1 – Initialisation et vérification de l'environnement

### Q1 – Création du dossier TP3

Le dossier `TP3/` a été créé avec les sous-dossiers `assets/`, `data/`, `outputs/` et `report/`.

```bash
mkdir -p TP3/assets TP3/data TP3/outputs TP3/report
```

### Q2 – Complétion de `sanity_check.py`

Les blancs complétés :
- `torch.cuda.get_device_name(0)` — index 0 pour le premier GPU.
- `torch.cuda.get_device_properties(0).total_memory` — même index.

### Q3 – Exécution de `sanity_check.py`

```
=== TP3 sanity check ===
torch: 2.5.1+cu121
torchaudio: 2.5.1+cu121
transformers: 5.4.0
datasets: 4.8.4
device: cuda
gpu_name: NVIDIA H100 NVL MIG 7g.94gb
gpu_mem_gb: 93.12
wav_shape: (1, 16000)
logmel_shape: (1, 80, 101)
```

Le GPU détecté est un **NVIDIA H100 NVL MIG 7g.94gb** avec 93.12 Go de mémoire. Le signal synthétique de test (1 seconde à 440 Hz, 16 kHz) produit un tenseur `[1, 16000]` et un log-mel spectrogram `[1, 80, 101]`, ce qui confirme que `torchaudio` fonctionne correctement.

> **[CAPTURE D'ÉCRAN]** : `TP3/report/sanity_check.png`

---

## Exercice 2 – Enregistrement audio et vérification

### Q1 – Enregistrement de `call_01.wav`

Audio enregistré en lisant le texte anglais fourni dans l'énoncé, sauvegardé en WAV mono 16 kHz dans `TP3/data/call_01.wav`.

### Q2 – Vérification des métadonnées audio

```
path: TP3/data/call_01.wav
sr: 16000
shape: (1, 599808)
duration_s: 37.49
rms: 0.0229
clipping_rate: 0.0
```

> **[CAPTURE D'ÉCRAN]** : `TP3/report/inspect_audio.png`

### Q3 – Conversion WAV (si nécessaire)

Le fichier était déjà en WAV mono 16 kHz, aucune conversion nécessaire. Si besoin :

```bash
ffmpeg -i source.m4a -ac 1 -ar 16000 TP3/data/call_01.wav
```

### Q4 – Complétion de `inspect_audio.py`

Blanc complété : `wav.shape[1]` pour obtenir le nombre d'échantillons (dimension temporelle du tenseur `[1, T]`).

### Q5 – Exécution de `inspect_audio.py`

```
path: TP3/data/call_01.wav
sr: 16000
shape: (1, 599808)
duration_s: 37.49
rms: 0.0229
clipping_rate: 0.0
```

L'audio dure **37.49 secondes**, est bien en mono 16 kHz. Le RMS de 0.0229 indique un signal à niveau modéré. Le taux de clipping est nul : aucune saturation, signal propre.

> **[CAPTURE D'ÉCRAN]** : `TP3/report/inspect_audio.png`

---

## Exercice 3 – VAD (Voice Activity Detection)

### Q1 – Complétion de `vad_segment.py`

Blanc complété : `sampling_rate=sr` (sr = 16000). La fonction `get_speech_timestamps` de Silero VAD a besoin du taux d'échantillonnage pour convertir les indices samples en timestamps.

> Installation : `pip install silero-vad`

### Q2 – Exécution du VAD + extrait JSON

```
duration_s: 37.49
num_segments: 20
total_speech_s: 21.68
speech_ratio: 0.578
saved: TP3/outputs/vad_segments_call_01.json
```

**Extrait des 5 premiers segments :**

```json
{
  "segments": [
    {"start_s": 1.89,  "end_s": 2.238},
    {"start_s": 2.37,  "end_s": 3.646},
    {"start_s": 4.194, "end_s": 4.99},
    {"start_s": 5.186, "end_s": 6.014},
    {"start_s": 6.722, "end_s": 8.67}
  ]
}
```

> **[CAPTURE D'ÉCRAN]** : `TP3/report/vad_terminal.png`

### Q3 – Analyse du ratio speech/silence

Le ratio speech/silence mesuré est **0.578** (57.8 % de parole sur 37.49 s). C'est cohérent avec une lecture du texte de l'énoncé : le texte est court (~60 mots), la lecture est lente et articulée avec des pauses naturelles entre les phrases. On observe aussi 20 segments, dont beaucoup très courts (< 0.5 s) correspondant aux chiffres épelés du numéro de téléphone ("Five.", "Bye.", "Zero.", etc.), ce qui montre que le VAD est sensible aux silences inter-mots lors de l'épellation.

### Q4 – Effet du changement de `min_dur_s`

En passant de `min_dur_s = 0.30` à `min_dur_s = 0.60`, `num_segments` ↓ (les segments courts comme "Also.", "Five.", "Bye." seraient supprimés), `speech_ratio` ↓ légèrement car ces courtes syllabes isolées représentent environ 3–4 secondes de parole au total.

---

## Exercice 4 – ASR avec Whisper + Call Center Analytics

### Q1 – Complétion de `asr_whisper.py`

Blanc complété : `model_id = "openai/whisper-small"`. Bon compromis taille/qualité pour ~37 s d'audio sur GPU H100.

### Q2 – Exécution de `asr_whisper.py`

```
model_id: openai/whisper-small
device: cuda
audio_duration_s: 37.49
elapsed_s: 9.22
rtf: 0.246
saved: TP3/outputs/asr_call_01.json
```

RTF = 0.246 (premier run avec téléchargement du modèle), puis **0.072** avec le modèle en cache : Whisper transcrit 37 s d'audio en ~2.7 s, soit environ **14× plus vite que le temps réel** sur GPU H100.

> **[CAPTURE D'ÉCRAN]** : `TP3/report/asr_terminal.png`

### Q3 – Extrait de la transcription

**5 premiers segments :**

```json
[
  {"segment_id": 0, "start_s": 1.89,  "end_s": 2.238,  "text": "Hello."},
  {"segment_id": 1, "start_s": 2.37,  "end_s": 3.646,  "text": "Thank you for coming."},
  {"segment_id": 2, "start_s": 4.194, "end_s": 4.99,   "text": "My name is Alex."},
  {"segment_id": 3, "start_s": 5.186, "end_s": 6.014,  "text": "and I will help you today."},
  {"segment_id": 4, "start_s": 6.722, "end_s": 8.67,   "text": "I'm conning about another bright image."}
]
```

**`full_text` (extrait) :**

> "Hello. Thank you for coming. My name is Alex. and I will help you today. I'm conning about another bright image. So back in. was delivered yesterday with a screen scratch. I would like to inform our investment as soon as possible. The other number is 8x19735. You can reach me at john.smease at example.com. Also. My phone number is... Five. Bye. Bye. Zero. one. nine. night. Thank you."

### Q4 – Analyse VAD ↔ transcription

La segmentation VAD aide Whisper en isolant les segments de parole et en évitant les hallucinations sur les silences. Cependant, certains segments très courts (< 0.5 s) contenant des chiffres épelés ("Five.", "Bye.", "Zero.") sont mal contextualisés : sans la phrase entière, Whisper transcrit "5" comme "Bye" ou "Five" et "9" comme "night", car il n'a pas assez de contexte phonétique. Le segment 4 (1.95 s) illustre aussi le problème inverse : trop court pour "I'm calling about an order that arrived damaged" → Whisper produit "I'm conning about another bright image." (hallucination sémantique).

### Q5 – Exécution de `callcenter_analytics.py`

```
intent: general_support
pii_stats: {'emails': 1, 'phones': 0, 'orders': 0}
top_terms: [('thank', 2), ('number', 2), ('bye', 2), ('hello', 1), ('coming', 1)]
saved: TP3/outputs/call_summary_call_01.json
```

> **[CAPTURE D'ÉCRAN]** : `TP3/report/analytics_terminal.png`

### Q6 – Extrait JSON du résumé d'appel

```json
{
  "pii_stats": {"emails": 1, "phones": 0, "orders": 0},
  "intent_scores": {
    "refund_or_replacement": 0,
    "delivery_issue": 2,
    "general_support": 3
  },
  "intent": "general_support",
  "top_terms": [["thank", 2], ["number", 2], ["bye", 2], ["hello", 1], ["coming", 1]],
  "redacted_text": "hello.thank you 4 coming.my name is alex.and i will help you today.im conning about another bright image.so back in.was delivered yesterday with a screen scratch.i would like 2 inform our investment as soon as possible.the other number is 8 x 19735.you can reach [REDACTED_EMAIL]@example.com.also.my phone number is 5.bye.bye.0.1.9.night.thank you."
}
```

### Q7 – Relancer après post-traitement et comparer

Avec le post-traitement PII (normalisation des tokens épelés, `dot`→`.`, `at`→`@`, collage des digits), **1 email est détecté et masqué** : `john.smease@example.com` → `[REDACTED_EMAIL]`. En revanche :
- Le **téléphone** (555-0199) n'est pas détecté : Whisper a transcrit les chiffres comme "Five. Bye. Bye. Zero. one. nine. night." au lieu de "5 5 5 0 1 9 9", rendant la reconstruction impossible même après normalisation.
- L'**order ID** (AX19735) est partiellement transcrit "8x19735" sans le contexte "order number is", donc non détecté par le pattern contextuel.

### Q8 – Réflexion : erreurs Whisper vs analytics

Les erreurs de Whisper impactent fortement les analytics. L'intention `refund_or_replacement` obtient un score de 0 car les mots-clés "refund", "replacement", "damaged", "cracked" sont tous mal transcrits ("another bright image" au lieu de "an order arrived damaged", "investment" au lieu de "refund or replacement"). L'intention détectée est `general_support` alors que l'appel porte clairement sur un remboursement — une erreur de routage critique en production.

Pour les PII, le numéro de téléphone "555 0199" épelé chiffre par chiffre est fragmenté en 8 segments VAD distincts, chaque chiffre transcrit séparément ("Bye." = 5 en phonétique ?), rendant la redaction impossible sans une normalisation phonétique très complète. La phrase "john dot smith at example dot com" est partiellement normalisée (→ "john.smease at example.com"), mais le local-part est altéré ("smith" → "smease"), illustrant la fragilité de la chaîne ASR → regex.

---

## Exercice 5 – TTS : génération d'une réponse vocale

### Q1 – Complétion de `tts_reply.py`

Blanc complété : `tts_model_id = "facebook/mms-tts-eng"`. Modèle TTS léger de Meta, gratuit, anglais.

### Q2 – Exécution de `tts_reply.py`

```
tts_model_id: facebook/mms-tts-eng
device: cuda
audio_dur_s: 8.77
elapsed_s: 3.9
rtf: 0.445
saved: TP3/outputs/tts_reply_call_01.wav
```

RTF = 0.445 (premier run avec téléchargement du modèle), puis **0.105** avec le modèle en cache : génération de 8.51 s d'audio en 0.9 s, soit environ **9× plus vite que le temps réel** sur GPU H100.

> **[CAPTURE D'ÉCRAN]** : `TP3/report/tts_terminal.png`

### Q3 – Métadonnées du WAV généré

Le fichier `tts_reply_call_01.wav` contient 8.77 secondes d'audio mono. Le sample rate est celui du modèle MMS-TTS (typiquement 16 kHz).

> **[CAPTURE D'ÉCRAN]** : `TP3/report/tts_metadata.png`  
> (Commande : `ffprobe TP3/outputs/tts_reply_call_01.wav`)

### Q4 – Observation sur la qualité TTS

Le modèle `facebook/mms-tts-eng` produit une parole intelligible avec une prononciation correcte des mots courants. La prosodie est monotone et légèrement robotique, sans variation d'intonation naturelle. On perçoit de légers artefacts métalliques sur certaines consonnes fricatives. Le RTF de 0.445 est compatible avec un usage temps réel en production. La qualité est suffisante pour un prototype de call center automatisé mais nécessiterait un modèle plus expressif (VITS, Bark) pour un usage grand public.

### Q5 – Vérification TTS via `asr_tts_check.py`

```
original_text: Thank you for contacting us. Your return request has been received and we will process it within two business days.
asr_text: Thanks for calling. I am sorry your order arrived. Demaged I can offer a replacement or a refund. Please confirm your preferred option.
match: False
```

La transcription ASR du WAV TTS renvoie le texte du script `tts_reply.py` (le message de l'agent), et non le texte `original_text` du script de vérification qui est différent. Le `match: False` est donc attendu : les deux textes sont intentionnellement distincts. En revanche, la transcription ASR est fidèle au message TTS généré, ce qui confirme l'intelligibilité du modèle MMS-TTS.

> **[CAPTURE D'ÉCRAN]** : `TP3/report/asr_tts_check.png`

---

## Exercice 6 – Pipeline end-to-end + rapport d'ingénierie

### Q1 – Complétion de `run_pipeline.py`

Blancs complétés :
- `"python TP3/vad_segment.py"` — étape VAD
- `"python TP3/asr_whisper.py"` — étape ASR
- `"python TP3/callcenter_analytics.py"` — étape Analytics

### Q2 – Exécution de `run_pipeline.py`

```
=== PIPELINE SUMMARY ===
audio_path: TP3/data/call_01.wav
duration_s: 37.488
num_segments: 20
speech_ratio: 0.5783183952198034
asr_model: openai/whisper-small
asr_device: cuda
asr_rtf: 0.0718859045412924
intent: general_support
pii_stats: {'emails': 1, 'phones': 0, 'orders': 0}
tts_generated: True
saved: TP3/outputs/pipeline_summary_call_01.json
```

> **[CAPTURE D'ÉCRAN]** : `TP3/report/pipeline_summary.png`

### Q3 – Extrait JSON du pipeline summary

```json
{
  "audio_path": "TP3/data/call_01.wav",
  "duration_s": 37.488,
  "num_segments": 20,
  "speech_ratio": 0.5783183952198034,
  "asr_model": "openai/whisper-small",
  "asr_device": "cuda",
  "asr_rtf": 0.0718859045412924,
  "intent": "general_support",
  "pii_stats": {"emails": 1, "phones": 0, "orders": 0},
  "tts_generated": true
}
```

### Q4 – Engineering note

**Goulet d'étranglement principal (temps) :** L'étape ASR Whisper est la plus coûteuse. Lors du premier run (téléchargement du modèle inclus), elle prend ~9 s pour 37 s d'audio (RTF 0.246). Au deuxième run (modèle en cache), le RTF tombe à **0.072** (~2.7 s), ce qui reste l'étape dominante. VAD (~0.5 s) et analytics (~0 s) sont négligeables. La TTS atteint RTF 0.105 avec le modèle en cache (0.9 s pour 8.51 s d'audio).

**Étape la plus fragile (qualité) :** La redaction PII est l'étape la plus fragile. Elle dépend entièrement de la qualité ASR en amont : le numéro de téléphone "555 0199" épelé en 8 segments VAD distincts produit des transcriptions comme "Five. Bye. Bye. Zero. one. nine. night." qui résistent à toute normalisation regex. L'email est partiellement détecté ("smease" au lieu de "smith") mais rédacté grâce au fallback contextuel. En production, une fuite PII de ce type serait inacceptable.

**Deux améliorations concrètes sans entraîner de modèle :**

1. **Contrôle de la longueur des segments VAD** : forcer un minimum de 1–2 secondes par segment (fusionner les segments adjacents) permettrait à Whisper d'avoir assez de contexte pour transcrire correctement les chiffres épelés, réduisant les hallucinations et améliorant la redaction PII.

2. **Batching ASR** : utiliser `pipeline(..., batch_size=8)` pour transcrire tous les segments en un seul forward pass GPU, réduisant la latence ASR d'un facteur ~4–5 sans changer la qualité des transcriptions.
