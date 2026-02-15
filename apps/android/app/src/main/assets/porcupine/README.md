# Porcupine Wake Word Models

Place your trained Porcupine models here:

1. **Keyword file** (`.ppn`) — trained wake word model (e.g., `moru_android.ppn`)
2. **Language model** (`.pv`) — Porcupine language params (e.g., `porcupine_params_es.pv`)

## How to generate

1. Go to [Picovoice Console](https://console.picovoice.ai/)
2. Sign up / log in (free tier available)
3. Copy your **AccessKey** from the dashboard
4. Go to **Porcupine** → **Custom Keywords**
5. Enter `Moru` as the wake word
6. Select **Spanish** as the language
7. Select **Android** as the platform
8. Download the `.ppn` file and place it in this folder
9. Download the Spanish language model (`porcupine_params_es.pv`) from:
   https://github.com/Picovoice/porcupine/tree/master/lib/common
   and place it in this folder

## Configuration in OpenClaw Android

In the app Settings → Voice → Porcupine section:

- **AccessKey**: Your Picovoice AccessKey
- **Keyword Path**: `porcupine/moru_android.ppn` (relative to assets)
- **Model Path**: `porcupine/porcupine_params_es.pv` (relative to assets)
- **Sensitivity**: `0.7` (0.0 = fewer false positives, 1.0 = fewer misses)
