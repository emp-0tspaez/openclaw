package ai.openclaw.android.voice

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.MediaPlayer
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import androidx.core.content.ContextCompat
import ai.openclaw.android.gateway.GatewaySession
import ai.openclaw.android.isCanonicalMainSessionKey
import ai.openclaw.android.normalizeMainKey
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlin.math.max

class TalkModeManager(
  private val context: Context,
  private val scope: CoroutineScope,
  private val session: GatewaySession,
  private val supportsChatSubscribe: Boolean,
  private val isConnected: () -> Boolean,
) {
  companion object {
    private const val tag = "TalkMode"
    private const val defaultModelFallback = "gpt-4o-mini-tts-2025-12-15"
    private const val defaultVoiceFallback = "onyx"
    private const val defaultOutputFormatFallback = "mp3"
    private const val OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"
  }

  private val mainHandler = Handler(Looper.getMainLooper())
  private val json = Json { ignoreUnknownKeys = true }

  private val _isEnabled = MutableStateFlow(false)
  val isEnabled: StateFlow<Boolean> = _isEnabled

  private val _isListening = MutableStateFlow(false)
  val isListening: StateFlow<Boolean> = _isListening

  private val _isSpeaking = MutableStateFlow(false)
  val isSpeaking: StateFlow<Boolean> = _isSpeaking

  private val _statusText = MutableStateFlow("Off")
  val statusText: StateFlow<String> = _statusText

  private val _lastAssistantText = MutableStateFlow<String?>(null)
  val lastAssistantText: StateFlow<String?> = _lastAssistantText

  private val _usingFallbackTts = MutableStateFlow(false)
  val usingFallbackTts: StateFlow<Boolean> = _usingFallbackTts

  private var recognizer: SpeechRecognizer? = null
  private var restartJob: Job? = null
  private var stopRequested = false
  private var listeningMode = false

  private var silenceJob: Job? = null
  private val silenceWindowMs = 700L
  private var lastTranscript: String = ""
  private var lastHeardAtMs: Long? = null
  private var lastSpokenText: String? = null
  private var lastInterruptedAtSeconds: Double? = null

  private var defaultVoice: String? = null
  private var currentVoice: String? = null
  private var defaultModel: String? = null
  private var currentModel: String? = null
  private var defaultOutputFormat: String? = null
  private var openaiApiKey: String? = null
  private var interruptOnSpeech: Boolean = true
  private var voiceOverrideActive = false
  private var modelOverrideActive = false
  private var mainSessionKey: String = "main"

  private var pendingRunId: String? = null
  private var pendingFinal: CompletableDeferred<Boolean>? = null
  private var chatSubscribedSessionKey: String? = null

  private var player: MediaPlayer? = null
  private var streamingSource: StreamingMediaDataSource? = null
  private var pcmTrack: AudioTrack? = null
  @Volatile private var pcmStopRequested = false
  private var systemTts: TextToSpeech? = null
  private var systemTtsPending: CompletableDeferred<Unit>? = null
  private var systemTtsPendingId: String? = null

  fun setMainSessionKey(sessionKey: String?) {
    val trimmed = sessionKey?.trim().orEmpty()
    if (trimmed.isEmpty()) return
    if (isCanonicalMainSessionKey(mainSessionKey)) return
    mainSessionKey = trimmed
  }

  fun setEnabled(enabled: Boolean) {
    if (_isEnabled.value == enabled) return
    _isEnabled.value = enabled
    if (enabled) {
      Log.d(tag, "enabled")
      start()
    } else {
      Log.d(tag, "disabled")
      stop()
    }
  }

  fun handleGatewayEvent(event: String, payloadJson: String?) {
    if (event != "chat") return
    if (payloadJson.isNullOrBlank()) return
    val pending = pendingRunId ?: return
    val obj =
      try {
        json.parseToJsonElement(payloadJson).asObjectOrNull()
      } catch (_: Throwable) {
        null
      } ?: return
    val runId = obj["runId"].asStringOrNull() ?: return
    if (runId != pending) return
    val state = obj["state"].asStringOrNull() ?: return
    if (state == "final") {
      pendingFinal?.complete(true)
      pendingFinal = null
      pendingRunId = null
    }
  }

  private fun start() {
    mainHandler.post {
      if (_isListening.value) return@post
      stopRequested = false
      listeningMode = true
      Log.d(tag, "start")

      if (!SpeechRecognizer.isRecognitionAvailable(context)) {
        _statusText.value = "Speech recognizer unavailable"
        Log.w(tag, "speech recognizer unavailable")
        return@post
      }

      val micOk =
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
          PackageManager.PERMISSION_GRANTED
      if (!micOk) {
        _statusText.value = "Microphone permission required"
        Log.w(tag, "microphone permission required")
        return@post
      }

      try {
        recognizer?.destroy()
        recognizer = SpeechRecognizer.createSpeechRecognizer(context).also { it.setRecognitionListener(listener) }
        startListeningInternal(markListening = true)
        startSilenceMonitor()
        Log.d(tag, "listening")
      } catch (err: Throwable) {
        _statusText.value = "Start failed: ${err.message ?: err::class.simpleName}"
        Log.w(tag, "start failed: ${err.message ?: err::class.simpleName}")
      }
    }
  }

  private fun stop() {
    stopRequested = true
    listeningMode = false
    restartJob?.cancel()
    restartJob = null
    silenceJob?.cancel()
    silenceJob = null
    lastTranscript = ""
    lastHeardAtMs = null
    _isListening.value = false
    _statusText.value = "Off"
    stopSpeaking()
    _usingFallbackTts.value = false
    chatSubscribedSessionKey = null

    mainHandler.post {
      recognizer?.cancel()
      recognizer?.destroy()
      recognizer = null
    }
    systemTts?.stop()
    systemTtsPending?.cancel()
    systemTtsPending = null
    systemTtsPendingId = null
  }

  private fun startListeningInternal(markListening: Boolean) {
    val r = recognizer ?: return
    val intent =
      Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
        putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.packageName)
      }

    if (markListening) {
      _statusText.value = "Listening"
      _isListening.value = true
    }
    r.startListening(intent)
  }

  private fun scheduleRestart(delayMs: Long = 350) {
    if (stopRequested) return
    restartJob?.cancel()
    restartJob =
      scope.launch {
        delay(delayMs)
        mainHandler.post {
          if (stopRequested) return@post
          try {
            // If the recognizer was destroyed (e.g. ERROR_SERVER_DISCONNECTED),
            // recreate it before restarting.
            if (recognizer == null && SpeechRecognizer.isRecognitionAvailable(context)) {
              recognizer = SpeechRecognizer.createSpeechRecognizer(context)
                .also { it.setRecognitionListener(listener) }
              Log.d(tag, "recognizer recreated after disconnect")
            }
            recognizer?.cancel()
            val shouldListen = listeningMode
            val shouldInterrupt = _isSpeaking.value && interruptOnSpeech
            if (!shouldListen && !shouldInterrupt) return@post
            startListeningInternal(markListening = shouldListen)
          } catch (_: Throwable) {
            // handled by onError
          }
        }
      }
  }

  private fun handleTranscript(text: String, isFinal: Boolean) {
    val trimmed = text.trim()
    if (_isSpeaking.value && interruptOnSpeech) {
      if (shouldInterrupt(trimmed)) {
        stopSpeaking()
      }
      return
    }

    if (!_isListening.value) return

    if (trimmed.isNotEmpty()) {
      lastTranscript = trimmed
      lastHeardAtMs = SystemClock.elapsedRealtime()
    }

    if (isFinal) {
      lastTranscript = trimmed
    }
  }

  private fun startSilenceMonitor() {
    silenceJob?.cancel()
    silenceJob =
      scope.launch {
        while (_isEnabled.value) {
          delay(200)
          checkSilence()
        }
      }
  }

  private fun checkSilence() {
    if (!_isListening.value) return
    val transcript = lastTranscript.trim()
    if (transcript.isEmpty()) return
    val lastHeard = lastHeardAtMs ?: return
    val elapsed = SystemClock.elapsedRealtime() - lastHeard
    if (elapsed < silenceWindowMs) return
    scope.launch { finalizeTranscript(transcript) }
  }

  private suspend fun finalizeTranscript(transcript: String) {
    listeningMode = false
    _isListening.value = false
    _statusText.value = "Thinking…"
    lastTranscript = ""
    lastHeardAtMs = null

    reloadConfig()
    val prompt = buildPrompt(transcript)
    if (!isConnected()) {
      _statusText.value = "Gateway not connected"
      Log.w(tag, "finalize: gateway not connected")
      start()
      return
    }

    try {
      val startedAt = System.currentTimeMillis().toDouble() / 1000.0
      subscribeChatIfNeeded(session = session, sessionKey = mainSessionKey)
      Log.d(tag, "chat.send start sessionKey=${mainSessionKey.ifBlank { "main" }} chars=${prompt.length}")
      val runId = sendChat(prompt, session)
      Log.d(tag, "chat.send ok runId=$runId")
      val ok = waitForChatFinal(runId)
      if (!ok) {
        Log.w(tag, "chat final timeout runId=$runId; attempting history fallback")
      }
      val assistant = waitForAssistantText(session, startedAt, if (ok) 12_000 else 25_000)
      if (assistant.isNullOrBlank()) {
        _statusText.value = "No reply"
        Log.w(tag, "assistant text timeout runId=$runId")
        start()
        return
      }
      Log.d(tag, "assistant text ok chars=${assistant.length}")
      playAssistant(assistant)
    } catch (err: Throwable) {
      _statusText.value = "Talk failed: ${err.message ?: err::class.simpleName}"
      Log.w(tag, "finalize failed: ${err.message ?: err::class.simpleName}")
    }

    if (_isEnabled.value) {
      start()
    }
  }

  private suspend fun subscribeChatIfNeeded(session: GatewaySession, sessionKey: String) {
    if (!supportsChatSubscribe) return
    val key = sessionKey.trim()
    if (key.isEmpty()) return
    if (chatSubscribedSessionKey == key) return
    try {
      session.sendNodeEvent("chat.subscribe", """{"sessionKey":"$key"}""")
      chatSubscribedSessionKey = key
      Log.d(tag, "chat.subscribe ok sessionKey=$key")
    } catch (err: Throwable) {
      Log.w(tag, "chat.subscribe failed sessionKey=$key err=${err.message ?: err::class.java.simpleName}")
    }
  }

  private fun buildPrompt(transcript: String): String {
    val lines = mutableListOf<String>()
    lastInterruptedAtSeconds?.let {
      lines.add("[interrupted at ${"%.1f".format(it)}s]")
      lastInterruptedAtSeconds = null
    }
    lines.add(transcript)
    return lines.joinToString("\n")
  }

  private suspend fun sendChat(message: String, session: GatewaySession): String {
    val runId = UUID.randomUUID().toString()
    val params =
      buildJsonObject {
        put("sessionKey", JsonPrimitive(mainSessionKey.ifBlank { "main" }))
        put("message", JsonPrimitive(message))
        put("thinking", JsonPrimitive("low"))
        put("timeoutMs", JsonPrimitive(30_000))
        put("idempotencyKey", JsonPrimitive(runId))
      }
    val res = session.request("chat.send", params.toString())
    val parsed = parseRunId(res) ?: runId
    if (parsed != runId) {
      pendingRunId = parsed
    }
    return parsed
  }

  private suspend fun waitForChatFinal(runId: String): Boolean {
    pendingFinal?.cancel()
    val deferred = CompletableDeferred<Boolean>()
    pendingRunId = runId
    pendingFinal = deferred

    val result =
      withContext(Dispatchers.IO) {
        try {
          kotlinx.coroutines.withTimeout(120_000) { deferred.await() }
        } catch (_: Throwable) {
          false
        }
      }

    if (!result) {
      pendingFinal = null
      pendingRunId = null
    }
    return result
  }

  private suspend fun waitForAssistantText(
    session: GatewaySession,
    sinceSeconds: Double,
    timeoutMs: Long,
  ): String? {
    val deadline = SystemClock.elapsedRealtime() + timeoutMs
    while (SystemClock.elapsedRealtime() < deadline) {
      val text = fetchLatestAssistantText(session, sinceSeconds)
      if (!text.isNullOrBlank()) return text
      delay(300)
    }
    return null
  }

  private suspend fun fetchLatestAssistantText(
    session: GatewaySession,
    sinceSeconds: Double? = null,
  ): String? {
    val key = mainSessionKey.ifBlank { "main" }
    val res = session.request("chat.history", "{\"sessionKey\":\"$key\"}")
    val root = json.parseToJsonElement(res).asObjectOrNull() ?: return null
    val messages = root["messages"] as? JsonArray ?: return null
    for (item in messages.reversed()) {
      val obj = item.asObjectOrNull() ?: continue
      if (obj["role"].asStringOrNull() != "assistant") continue
      if (sinceSeconds != null) {
        val timestamp = obj["timestamp"].asDoubleOrNull()
        if (timestamp != null && !TalkModeRuntime.isMessageTimestampAfter(timestamp, sinceSeconds)) continue
      }
      val content = obj["content"] as? JsonArray ?: continue
      val text =
        content.mapNotNull { entry ->
          entry.asObjectOrNull()?.get("text")?.asStringOrNull()?.trim()
        }.filter { it.isNotEmpty() }
      if (text.isNotEmpty()) return text.joinToString("\n")
    }
    return null
  }

  private suspend fun playAssistant(text: String) {
    val parsed = TalkDirectiveParser.parse(text)
    if (parsed.unknownKeys.isNotEmpty()) {
      Log.w(tag, "Unknown talk directive keys: ${parsed.unknownKeys}")
    }
    val directive = parsed.directive
    val cleaned = parsed.stripped.trim()
    if (cleaned.isEmpty()) return
    _lastAssistantText.value = cleaned

    _statusText.value = "Speaking…"
    _isSpeaking.value = true
    lastSpokenText = cleaned
    ensureInterruptListener()

    val apiKey = openaiApiKey?.trim()?.takeIf { it.isNotEmpty() }
      ?: System.getenv("OPENAI_API_KEY")?.trim()

    try {
      if (apiKey.isNullOrEmpty()) {
        Log.w(tag, "missing OPENAI_API_KEY; falling back to system voice")
        _usingFallbackTts.value = true
        _statusText.value = "Speaking (System)…"
        speakWithSystemTts(cleaned)
      } else {
        _usingFallbackTts.value = false
        val ttsStarted = SystemClock.elapsedRealtime()
        val voice = directive?.voiceId?.trim()?.takeIf { it.isNotEmpty() } ?: currentVoice ?: defaultVoice ?: defaultVoiceFallback
        val model = directive?.modelId?.trim()?.takeIf { it.isNotEmpty() } ?: currentModel ?: defaultModel ?: defaultModelFallback
        val speed = TalkModeRuntime.resolveSpeed(directive?.speed, directive?.rateWpm) ?: 1.0
        val request = OpenAiTtsRequest(
          input = cleaned,
          model = model,
          voice = voice,
          responseFormat = defaultOutputFormat ?: defaultOutputFormatFallback,
          speed = speed,
        )
        streamAndPlayOpenAi(apiKey = apiKey, request = request)
        Log.d(tag, "openai tts ok durMs=${SystemClock.elapsedRealtime() - ttsStarted}")
      }
    } catch (err: Throwable) {
      Log.w(tag, "openai tts failed: ${err.message ?: err::class.simpleName}; falling back to system voice")
      try {
        _usingFallbackTts.value = true
        _statusText.value = "Speaking (System)…"
        speakWithSystemTts(cleaned)
      } catch (fallbackErr: Throwable) {
        _statusText.value = "Speak failed: ${fallbackErr.message ?: fallbackErr::class.simpleName}"
        Log.w(tag, "system voice failed: ${fallbackErr.message ?: fallbackErr::class.simpleName}")
      }
    }

    _isSpeaking.value = false
  }

  private suspend fun streamAndPlayOpenAi(apiKey: String, request: OpenAiTtsRequest) {
    stopSpeaking(resetInterrupt = false)
    pcmStopRequested = false

    val fmt = request.responseFormat.lowercase()
    if (fmt == "pcm") {
      try {
        streamAndPlayPcmOpenAi(apiKey = apiKey, request = request, sampleRate = 24000)
        return
      } catch (err: Throwable) {
        if (pcmStopRequested) return
        Log.w(tag, "pcm playback failed; falling back to mp3: ${err.message ?: err::class.simpleName}")
      }
      // retry as mp3
      streamAndPlayMp3OpenAi(apiKey = apiKey, request = request.copy(responseFormat = "mp3"))
    } else {
      streamAndPlayMp3OpenAi(apiKey = apiKey, request = request)
    }
  }

  private suspend fun streamAndPlayMp3OpenAi(apiKey: String, request: OpenAiTtsRequest) {
    val dataSource = StreamingMediaDataSource()
    streamingSource = dataSource

    val player = MediaPlayer()
    this.player = player

    val prepared = CompletableDeferred<Unit>()
    val finished = CompletableDeferred<Unit>()

    player.setAudioAttributes(
      AudioAttributes.Builder()
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .setUsage(AudioAttributes.USAGE_MEDIA)
        .build(),
    )
    player.setOnPreparedListener {
      it.start()
      prepared.complete(Unit)
    }
    player.setOnCompletionListener {
      finished.complete(Unit)
    }
    player.setOnErrorListener { _, _, _ ->
      finished.completeExceptionally(IllegalStateException("MediaPlayer error"))
      true
    }

    player.setDataSource(dataSource)
    withContext(Dispatchers.Main) {
      player.prepareAsync()
    }

    val fetchError = CompletableDeferred<Throwable?>()
    val fetchJob =
      scope.launch(Dispatchers.IO) {
        try {
          streamOpenAiTts(apiKey = apiKey, request = request, sink = dataSource)
          fetchError.complete(null)
        } catch (err: Throwable) {
          dataSource.fail()
          fetchError.complete(err)
        }
      }

    Log.d(tag, "openai mp3 play start")
    try {
      prepared.await()
      finished.await()
      fetchError.await()?.let { throw it }
    } finally {
      fetchJob.cancel()
      cleanupPlayer()
    }
    Log.d(tag, "openai mp3 play done")
  }

  private suspend fun streamAndPlayPcmOpenAi(
    apiKey: String,
    request: OpenAiTtsRequest,
    sampleRate: Int,
  ) {
    val minBuffer =
      AudioTrack.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_OUT_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
      )
    if (minBuffer <= 0) {
      throw IllegalStateException("AudioTrack buffer size invalid: $minBuffer")
    }

    val bufferSize = max(minBuffer * 2, 8 * 1024)
    val track =
      AudioTrack(
        AudioAttributes.Builder()
          .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
          .setUsage(AudioAttributes.USAGE_MEDIA)
          .build(),
        AudioFormat.Builder()
          .setSampleRate(sampleRate)
          .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
          .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
          .build(),
        bufferSize,
        AudioTrack.MODE_STREAM,
        AudioManager.AUDIO_SESSION_ID_GENERATE,
      )
    if (track.state != AudioTrack.STATE_INITIALIZED) {
      track.release()
      throw IllegalStateException("AudioTrack init failed")
    }
    pcmTrack = track
    track.play()

    Log.d(tag, "openai pcm play start sampleRate=$sampleRate")
    try {
      streamOpenAiPcm(apiKey = apiKey, request = request.copy(responseFormat = "pcm"), track = track)
    } finally {
      cleanupPcmTrack()
    }
    Log.d(tag, "openai pcm play done")
  }

  private suspend fun speakWithSystemTts(text: String) {
    val trimmed = text.trim()
    if (trimmed.isEmpty()) return
    val ok = ensureSystemTts()
    if (!ok) {
      throw IllegalStateException("system TTS unavailable")
    }

    val tts = systemTts ?: throw IllegalStateException("system TTS unavailable")
    val utteranceId = "talk-${UUID.randomUUID()}"
    val deferred = CompletableDeferred<Unit>()
    systemTtsPending?.cancel()
    systemTtsPending = deferred
    systemTtsPendingId = utteranceId

    withContext(Dispatchers.Main) {
      // Use STREAM_MUSIC to avoid Samsung USAGE_ASSISTANT suppression
      val params = Bundle().apply {
        putInt(TextToSpeech.Engine.KEY_PARAM_STREAM, AudioManager.STREAM_MUSIC)
      }
      tts.speak(trimmed, TextToSpeech.QUEUE_FLUSH, params, utteranceId)
    }

    withContext(Dispatchers.IO) {
      try {
        kotlinx.coroutines.withTimeout(180_000) { deferred.await() }
      } catch (err: Throwable) {
        throw err
      }
    }
  }

  private suspend fun ensureSystemTts(): Boolean {
    if (systemTts != null) return true
    return withContext(Dispatchers.Main) {
      val deferred = CompletableDeferred<Boolean>()
      val tts =
        try {
          TextToSpeech(context) { status ->
            deferred.complete(status == TextToSpeech.SUCCESS)
          }
        } catch (_: Throwable) {
          deferred.complete(false)
          null
        }
      if (tts == null) return@withContext false

      tts.setOnUtteranceProgressListener(
        object : UtteranceProgressListener() {
          override fun onStart(utteranceId: String?) {}

          override fun onDone(utteranceId: String?) {
            if (utteranceId == null) return
            if (utteranceId != systemTtsPendingId) return
            systemTtsPending?.complete(Unit)
            systemTtsPending = null
            systemTtsPendingId = null
          }

          @Suppress("OVERRIDE_DEPRECATION")
          @Deprecated("Deprecated in Java")
          override fun onError(utteranceId: String?) {
            if (utteranceId == null) return
            if (utteranceId != systemTtsPendingId) return
            systemTtsPending?.completeExceptionally(IllegalStateException("system TTS error"))
            systemTtsPending = null
            systemTtsPendingId = null
          }

          override fun onError(utteranceId: String?, errorCode: Int) {
            if (utteranceId == null) return
            if (utteranceId != systemTtsPendingId) return
            systemTtsPending?.completeExceptionally(IllegalStateException("system TTS error $errorCode"))
            systemTtsPending = null
            systemTtsPendingId = null
          }
        },
      )

      val ok =
        try {
          deferred.await()
        } catch (_: Throwable) {
          false
        }
      if (ok) {
        // Route system TTS through USAGE_MEDIA to bypass Samsung suppression
        tts.setAudioAttributes(
          AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()
        )
        systemTts = tts
      } else {
        tts.shutdown()
      }
      ok
    }
  }

  private fun stopSpeaking(resetInterrupt: Boolean = true) {
    pcmStopRequested = true
    if (!_isSpeaking.value) {
      cleanupPlayer()
      cleanupPcmTrack()
      systemTts?.stop()
      systemTtsPending?.cancel()
      systemTtsPending = null
      systemTtsPendingId = null
      return
    }
    if (resetInterrupt) {
      val currentMs = player?.currentPosition?.toDouble() ?: 0.0
      lastInterruptedAtSeconds = currentMs / 1000.0
    }
    cleanupPlayer()
    cleanupPcmTrack()
    systemTts?.stop()
    systemTtsPending?.cancel()
    systemTtsPending = null
    systemTtsPendingId = null
    _isSpeaking.value = false
  }

  private fun cleanupPlayer() {
    player?.stop()
    player?.release()
    player = null
    streamingSource?.close()
    streamingSource = null
  }

  private fun cleanupPcmTrack() {
    val track = pcmTrack ?: return
    try {
      track.pause()
      track.flush()
      track.stop()
    } catch (_: Throwable) {
      // ignore cleanup errors
    } finally {
      track.release()
    }
    pcmTrack = null
  }

  private fun shouldInterrupt(transcript: String): Boolean {
    val trimmed = transcript.trim()
    if (trimmed.length < 3) return false
    val spoken = lastSpokenText?.lowercase()
    if (spoken != null && spoken.contains(trimmed.lowercase())) return false
    return true
  }

  private suspend fun reloadConfig() {
    val envKey = System.getenv("OPENAI_API_KEY")?.trim()
    try {
      val res = session.request("talk.config", """{"includeSecrets":true}""")
      val root = json.parseToJsonElement(res).asObjectOrNull()
      val config = root?.get("config").asObjectOrNull()
      val talk = config?.get("talk").asObjectOrNull()
      val sessionCfg = config?.get("session").asObjectOrNull()
      val mainKey = normalizeMainKey(sessionCfg?.get("mainKey").asStringOrNull())
      val voice = (talk?.get("voiceId") ?: talk?.get("voice"))?.asStringOrNull()?.trim()?.takeIf { it.isNotEmpty() }
      val model = (talk?.get("modelId") ?: talk?.get("model"))?.asStringOrNull()?.trim()?.takeIf { it.isNotEmpty() }
      val outputFormat = (talk?.get("outputFormat") ?: talk?.get("responseFormat"))?.asStringOrNull()?.trim()?.takeIf { it.isNotEmpty() }
      val key = talk?.get("apiKey")?.asStringOrNull()?.trim()?.takeIf { it.isNotEmpty() }
      val interrupt = talk?.get("interruptOnSpeech")?.asBooleanOrNull()

      if (!isCanonicalMainSessionKey(mainSessionKey)) {
        mainSessionKey = mainKey
      }
      defaultVoice = voice ?: defaultVoiceFallback
      if (!voiceOverrideActive) currentVoice = defaultVoice
      defaultModel = model ?: defaultModelFallback
      if (!modelOverrideActive) currentModel = defaultModel
      defaultOutputFormat = outputFormat ?: defaultOutputFormatFallback
      openaiApiKey = key ?: envKey?.takeIf { it.isNotEmpty() }
      if (interrupt != null) interruptOnSpeech = interrupt
      Log.d(tag, "reloadConfig ok voice=$defaultVoice apiKey=${if (openaiApiKey.isNullOrEmpty()) "MISSING" else "present(${openaiApiKey!!.length})"} model=$defaultModel")
    } catch (err: Throwable) {
      Log.w(tag, "reloadConfig failed: ${err.message ?: err::class.simpleName}")
      defaultVoice = defaultVoiceFallback
      defaultModel = defaultModelFallback
      if (!modelOverrideActive) currentModel = defaultModel
      openaiApiKey = envKey?.takeIf { it.isNotEmpty() }
      defaultOutputFormat = defaultOutputFormatFallback
    }
  }

  private fun parseRunId(jsonString: String): String? {
    val obj = json.parseToJsonElement(jsonString).asObjectOrNull() ?: return null
    return obj["runId"].asStringOrNull()
  }

  private suspend fun streamOpenAiTts(
    apiKey: String,
    request: OpenAiTtsRequest,
    sink: StreamingMediaDataSource,
  ) {
    withContext(Dispatchers.IO) {
      val conn = openOpenAiTtsConnection(apiKey = apiKey)
      try {
        val payload = buildOpenAiPayload(request)
        conn.outputStream.use { it.write(payload.toByteArray()) }

        val code = conn.responseCode
        if (code >= 400) {
          val message = conn.errorStream?.readBytes()?.toString(Charsets.UTF_8) ?: ""
          sink.fail()
          throw IllegalStateException("OpenAI TTS failed: $code $message")
        }

        val buffer = ByteArray(8 * 1024)
        conn.inputStream.use { input ->
          while (true) {
            val read = input.read(buffer)
            if (read <= 0) break
            sink.append(buffer.copyOf(read))
          }
        }
        sink.finish()
      } finally {
        conn.disconnect()
      }
    }
  }

  private suspend fun streamOpenAiPcm(
    apiKey: String,
    request: OpenAiTtsRequest,
    track: AudioTrack,
  ) {
    withContext(Dispatchers.IO) {
      val conn = openOpenAiTtsConnection(apiKey = apiKey)
      try {
        val payload = buildOpenAiPayload(request)
        conn.outputStream.use { it.write(payload.toByteArray()) }

        val code = conn.responseCode
        if (code >= 400) {
          val message = conn.errorStream?.readBytes()?.toString(Charsets.UTF_8) ?: ""
          throw IllegalStateException("OpenAI TTS failed: $code $message")
        }

        val buffer = ByteArray(8 * 1024)
        conn.inputStream.use { input ->
          while (true) {
            if (pcmStopRequested) return@withContext
            val read = input.read(buffer)
            if (read <= 0) break
            var offset = 0
            while (offset < read) {
              if (pcmStopRequested) return@withContext
              val wrote =
                try {
                  track.write(buffer, offset, read - offset)
                } catch (err: Throwable) {
                  if (pcmStopRequested) return@withContext
                  throw err
                }
              if (wrote <= 0) {
                if (pcmStopRequested) return@withContext
                throw IllegalStateException("AudioTrack write failed: $wrote")
              }
              offset += wrote
            }
          }
        }
      } finally {
        conn.disconnect()
      }
    }
  }

  private fun openOpenAiTtsConnection(apiKey: String): HttpURLConnection {
    val url = URL(OPENAI_TTS_URL)
    val conn = url.openConnection() as HttpURLConnection
    conn.requestMethod = "POST"
    conn.connectTimeout = 30_000
    conn.readTimeout = 60_000
    conn.setRequestProperty("Content-Type", "application/json")
    conn.setRequestProperty("Authorization", "Bearer $apiKey")
    conn.doOutput = true
    return conn
  }

  private fun buildOpenAiPayload(request: OpenAiTtsRequest): String {
    val payload =
      buildJsonObject {
        put("model", JsonPrimitive(request.model))
        put("input", JsonPrimitive(request.input))
        put("voice", JsonPrimitive(request.voice))
        put("response_format", JsonPrimitive(request.responseFormat))
        if (request.speed != 1.0) {
          put("speed", JsonPrimitive(request.speed))
        }
      }
    return payload.toString()
  }

  private data class OpenAiTtsRequest(
    val input: String,
    val model: String,
    val voice: String,
    val responseFormat: String,
    val speed: Double = 1.0,
  )

  private object TalkModeRuntime {
    fun resolveSpeed(speed: Double?, rateWpm: Int?): Double? {
      if (rateWpm != null && rateWpm > 0) {
        val resolved = rateWpm.toDouble() / 175.0
        if (resolved < 0.25 || resolved > 4.0) return null
        return resolved
      }
      if (speed != null) {
        if (speed < 0.25 || speed > 4.0) return null
        return speed
      }
      return null
    }

    fun isMessageTimestampAfter(timestamp: Double, sinceSeconds: Double): Boolean {
      val sinceMs = sinceSeconds * 1000
      return if (timestamp > 10_000_000_000) {
        timestamp >= sinceMs - 500
      } else {
        timestamp >= sinceSeconds - 0.5
      }
    }
  }

  private fun ensureInterruptListener() {
    if (!interruptOnSpeech || !_isEnabled.value) return
    mainHandler.post {
      if (stopRequested) return@post
      if (!SpeechRecognizer.isRecognitionAvailable(context)) return@post
      try {
        if (recognizer == null) {
          recognizer = SpeechRecognizer.createSpeechRecognizer(context).also { it.setRecognitionListener(listener) }
        }
        recognizer?.cancel()
        startListeningInternal(markListening = false)
      } catch (_: Throwable) {
        // ignore
      }
    }
  }

  // OpenAI TTS voices: alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer
  private val validOpenAiVoices = setOf("alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer")

  private val listener =
    object : RecognitionListener {
      override fun onReadyForSpeech(params: Bundle?) {
        if (_isEnabled.value) {
          _statusText.value = if (_isListening.value) "Listening" else _statusText.value
        }
      }

      override fun onBeginningOfSpeech() {}

      override fun onRmsChanged(rmsdB: Float) {}

      override fun onBufferReceived(buffer: ByteArray?) {}

      override fun onEndOfSpeech() {
        scheduleRestart()
      }

      override fun onError(error: Int) {
        if (stopRequested) return
        _isListening.value = false
        if (error == SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS) {
          _statusText.value = "Microphone permission required"
          return
        }

        // ERROR_SERVER_DISCONNECTED = 11 (API 31+) — service connection lost;
        // must destroy & recreate the recognizer.
        val isServerDisconnect = error == 11
        if (isServerDisconnect) {
          Log.w(tag, "SpeechRecognizer service disconnected (error 11); recreating")
          mainHandler.post {
            try { recognizer?.destroy() } catch (_: Throwable) {}
            recognizer = null
          }
        }

        _statusText.value =
          when (error) {
            SpeechRecognizer.ERROR_AUDIO -> "Audio error"
            SpeechRecognizer.ERROR_CLIENT -> "Client error"
            SpeechRecognizer.ERROR_NETWORK -> "Network error"
            SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Network timeout"
            SpeechRecognizer.ERROR_NO_MATCH -> "Listening"
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognizer busy"
            SpeechRecognizer.ERROR_SERVER -> "Server error"
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Listening"
            11 -> "Reconnecting…"  // ERROR_SERVER_DISCONNECTED
            else -> "Speech error ($error)"
          }
        scheduleRestart(delayMs = if (isServerDisconnect) 1500 else 600)
      }

      override fun onResults(results: Bundle?) {
        val list = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION).orEmpty()
        list.firstOrNull()?.let { handleTranscript(it, isFinal = true) }
        scheduleRestart()
      }

      override fun onPartialResults(partialResults: Bundle?) {
        val list = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION).orEmpty()
        list.firstOrNull()?.let { handleTranscript(it, isFinal = false) }
      }

      override fun onEvent(eventType: Int, params: Bundle?) {}
    }
}

private fun JsonElement?.asObjectOrNull(): JsonObject? = this as? JsonObject

private fun JsonElement?.asStringOrNull(): String? =
  (this as? JsonPrimitive)?.takeIf { it.isString }?.content

private fun JsonElement?.asDoubleOrNull(): Double? {
  val primitive = this as? JsonPrimitive ?: return null
  return primitive.content.toDoubleOrNull()
}

private fun JsonElement?.asBooleanOrNull(): Boolean? {
  val primitive = this as? JsonPrimitive ?: return null
  val content = primitive.content.trim().lowercase()
  return when (content) {
    "true", "yes", "1" -> true
    "false", "no", "0" -> false
    else -> null
  }
}
