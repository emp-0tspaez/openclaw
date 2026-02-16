package ai.openclaw.android.voice

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/**
 * Voice wake manager powered by OpenWakeWord (ONNX).
 *
 * Phase 1 — OpenWakeWordDetector listens continuously via AudioRecord (no audio focus).
 * Phase 2 — When the wake word is detected, SpeechRecognizer runs briefly to capture the command.
 * Phase 3 — After command capture (or timeout), returns to Phase 1.
 *
 * Falls back to SpeechRecognizer-only if no ONNX models are available.
 */
class VoiceWakeManager(
  private val context: Context,
  private val scope: CoroutineScope,
  private val onCommand: suspend (String) -> Unit,
) {
  companion object {
    private const val TAG = "VoiceWakeManager"
    /** Duration (ms) to wait for speech after wake word detection. */
    private const val COMMAND_TIMEOUT_MS = 6_000L
  }

  private val mainHandler = Handler(Looper.getMainLooper())

  private val _isListening = MutableStateFlow(false)
  val isListening: StateFlow<Boolean> = _isListening

  private val _statusText = MutableStateFlow("Off")
  val statusText: StateFlow<String> = _statusText

  var triggerWords: List<String> = emptyList()
    private set

  // ── OpenWakeWord detector ─────────────────────────────────────────
  private var wakeWordDetector: OpenWakeWordDetector? = null
  private var detectorInitialized = false

  // ── Legacy / command‑capture SpeechRecognizer ─────────────────────
  private var recognizer: SpeechRecognizer? = null
  private var commandTimeoutJob: Job? = null

  // ── Configuration ─────────────────────────────────────────────────
  var threshold: Float = 0.5f
  var activeModels: List<String> = listOf("hey_jarvis_v0.1.onnx")

  private var stopRequested = false
  private var capturingCommand = false

  // ── Public API ────────────────────────────────────────────────────

  fun setTriggerWords(words: List<String>) {
    triggerWords = words
  }

  fun start() {
    mainHandler.post {
      if (_isListening.value) return@post
      stopRequested = false
      capturingCommand = false

      scope.launch(Dispatchers.IO) {
        val detector = getOrInitDetector()
        if (detector != null) {
          mainHandler.post { startWakeWordDetector(detector) }
        } else {
          Log.w(TAG, "OpenWakeWord init failed — falling back to SpeechRecognizer")
          mainHandler.post { startLegacy() }
        }
      }
    }
  }

  fun stop(statusText: String = "Off") {
    stopRequested = true
    commandTimeoutJob?.cancel()
    commandTimeoutJob = null
    mainHandler.post {
      _isListening.value = false
      _statusText.value = statusText
      capturingCommand = false
      stopDetectorInternal()
      stopRecognizerInternal()
    }
  }

  // ── OpenWakeWord engine ───────────────────────────────────────────

  private fun getOrInitDetector(): OpenWakeWordDetector? {
    if (detectorInitialized && wakeWordDetector != null) return wakeWordDetector

    try {
      val detector = OpenWakeWordDetector(context) {
        Log.d(TAG, "Wake word detected!")
        mainHandler.post {
          if (stopRequested) return@post
          _statusText.value = "Triggered"
          startCommandCapture()
        }
      }
      detector.threshold = threshold
      detector.activeModels = activeModels

      val ok = detector.initialize()
      if (!ok) {
        detector.release()
        return null
      }
      wakeWordDetector = detector
      detectorInitialized = true
      return detector
    } catch (err: Throwable) {
      Log.e(TAG, "OpenWakeWord init failed", err)
      return null
    }
  }

  private fun startWakeWordDetector(detector: OpenWakeWordDetector) {
    try {
      stopDetectorInternal()

      val ok = detector.start()
      if (ok) {
        wakeWordDetector = detector
        _isListening.value = true
        _statusText.value = "Listening"
        Log.d(TAG, "OpenWakeWord started")
      } else {
        _isListening.value = false
        _statusText.value = "Wake word start failed"
        Log.e(TAG, "OpenWakeWord start returned false")
      }
    } catch (err: Throwable) {
      _isListening.value = false
      _statusText.value = "Wake word error: ${err.message ?: err::class.simpleName}"
      Log.e(TAG, "OpenWakeWord start failed", err)
    }
  }

  private fun stopDetectorInternal() {
    try {
      wakeWordDetector?.stop()
    } catch (_: Throwable) { }
  }

  // ── Command capture (brief SpeechRecognizer after wake word) ──────

  private fun startCommandCapture() {
    capturingCommand = true

    if (!SpeechRecognizer.isRecognitionAvailable(context)) {
      Log.w(TAG, "SpeechRecognizer unavailable for command capture — sending empty wake trigger")
      dispatchWakeTrigger()
      return
    }

    try {
      // Pause detector while SpeechRecognizer is active (they share the mic).
      stopDetectorInternal()

      stopRecognizerInternal()
      recognizer = SpeechRecognizer.createSpeechRecognizer(context).also {
        it.setRecognitionListener(commandListener)
      }

      val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
        putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.packageName)
      }

      recognizer?.startListening(intent)
      _statusText.value = "Listening for command…"
      Log.d(TAG, "Command capture started")

      // Safety timeout — if no speech arrives, resume detector.
      commandTimeoutJob?.cancel()
      commandTimeoutJob = scope.launch {
        delay(COMMAND_TIMEOUT_MS)
        mainHandler.post {
          if (capturingCommand && !stopRequested) {
            Log.d(TAG, "Command capture timeout — resuming wake word detector")
            finishCommandCapture()
          }
        }
      }
    } catch (err: Throwable) {
      Log.w(TAG, "Command capture failed: ${err.message}")
      dispatchWakeTrigger()
    }
  }

  private fun finishCommandCapture() {
    capturingCommand = false
    commandTimeoutJob?.cancel()
    commandTimeoutJob = null
    stopRecognizerInternal()
    resumeDetector()
  }

  private fun resumeDetector() {
    if (stopRequested) return
    val detector = wakeWordDetector ?: return
    try {
      detector.start()
      _isListening.value = true
      _statusText.value = "Listening"
      Log.d(TAG, "Wake word detector resumed")
    } catch (err: Throwable) {
      _statusText.value = "Wake word resume error: ${err.message}"
      Log.e(TAG, "Wake word detector resume failed", err)
    }
  }

  private fun dispatchWakeTrigger() {
    // No command captured — send empty trigger so the agent knows the user said the wake word.
    scope.launch { onCommand("") }
    mainHandler.post { finishCommandCapture() }
  }

  private fun stopRecognizerInternal() {
    try {
      recognizer?.cancel()
      recognizer?.destroy()
    } catch (_: Throwable) { }
    recognizer = null
  }

  private val commandListener = object : RecognitionListener {
    override fun onReadyForSpeech(params: Bundle?) {
      _statusText.value = "Listening for command…"
    }

    override fun onBeginningOfSpeech() {}
    override fun onRmsChanged(rmsdB: Float) {}
    override fun onBufferReceived(buffer: ByteArray?) {}

    override fun onEndOfSpeech() {
      // Wait for onResults — don't restart yet.
    }

    override fun onError(error: Int) {
      if (stopRequested) return
      Log.w(TAG, "Command capture SpeechRecognizer error=$error")
      // Any error during command capture — just resume wake word detector.
      mainHandler.post { finishCommandCapture() }
    }

    override fun onResults(results: Bundle?) {
      val text = results
        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
        .orEmpty()
        .firstOrNull()
        ?.trim()
        .orEmpty()

      if (text.isNotEmpty()) {
        Log.d(TAG, "Command captured: $text")
        scope.launch { onCommand(text) }
      } else {
        Log.d(TAG, "Command capture empty result")
      }
      mainHandler.post { finishCommandCapture() }
    }

    override fun onPartialResults(partialResults: Bundle?) {
      // Could dispatch partial, but we wait for final results for accuracy.
    }

    override fun onEvent(eventType: Int, params: Bundle?) {}
  }

  // ── Legacy SpeechRecognizer‑only mode (backward compat) ───────────

  private var lastDispatched: String? = null

  private fun startLegacy() {
    if (!SpeechRecognizer.isRecognitionAvailable(context)) {
      _isListening.value = false
      _statusText.value = "Speech recognizer unavailable"
      return
    }

    try {
      stopRecognizerInternal()
      recognizer = SpeechRecognizer.createSpeechRecognizer(context).also {
        it.setRecognitionListener(legacyListener)
      }
      startLegacyListeningInternal()
    } catch (err: Throwable) {
      _isListening.value = false
      _statusText.value = "Start failed: ${err.message ?: err::class.simpleName}"
    }
  }

  private fun startLegacyListeningInternal() {
    val r = recognizer ?: return
    val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
      putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
      putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
      putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
      putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.packageName)
    }
    _statusText.value = "Listening"
    _isListening.value = true
    r.startListening(intent)
  }

  private fun scheduleLegacyRestart(delayMs: Long = 350) {
    if (stopRequested) return
    commandTimeoutJob?.cancel()
    commandTimeoutJob = scope.launch {
      delay(delayMs)
      mainHandler.post {
        if (stopRequested) return@post
        try {
          recognizer?.cancel()
          startLegacyListeningInternal()
        } catch (_: Throwable) { }
      }
    }
  }

  private fun handleLegacyTranscription(text: String) {
    val command = VoiceWakeCommandExtractor.extractCommand(text, triggerWords) ?: return
    if (command == lastDispatched) return
    lastDispatched = command
    scope.launch { onCommand(command) }
    _statusText.value = "Triggered"
    scheduleLegacyRestart(delayMs = 650)
  }

  private val legacyListener = object : RecognitionListener {
    override fun onReadyForSpeech(params: Bundle?) { _statusText.value = "Listening" }
    override fun onBeginningOfSpeech() {}
    override fun onRmsChanged(rmsdB: Float) {}
    override fun onBufferReceived(buffer: ByteArray?) {}
    override fun onEndOfSpeech() { scheduleLegacyRestart() }

    override fun onError(error: Int) {
      if (stopRequested) return
      _isListening.value = false
      if (error == SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS) {
        _statusText.value = "Microphone permission required"
        return
      }
      _statusText.value = when (error) {
        SpeechRecognizer.ERROR_NO_MATCH,
        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Listening"
        else -> "Speech error ($error)"
      }
      scheduleLegacyRestart(delayMs = 600)
    }

    override fun onResults(results: Bundle?) {
      results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
        .orEmpty()
        .firstOrNull()
        ?.let(::handleLegacyTranscription)
      scheduleLegacyRestart()
    }

    override fun onPartialResults(partialResults: Bundle?) {
      partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
        .orEmpty()
        .firstOrNull()
        ?.let(::handleLegacyTranscription)
    }

    override fun onEvent(eventType: Int, params: Bundle?) {}
  }
}
