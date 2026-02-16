package ai.openclaw.android.voice

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.core.content.ContextCompat
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/**
 * On-device wake word detector using openWakeWord ONNX models.
 *
 * Uses AudioRecord directly (no SpeechRecognizer) → does NOT steal audio focus
 * and does NOT interrupt media playback.
 *
 * Pipeline:
 *   AudioRecord (16kHz mono PCM) → melspectrogram.onnx → embedding_model.onnx → wake_word.onnx
 */
class OpenWakeWordDetector(
  private val context: Context,
  private val onWakeWordDetected: () -> Unit,
) {
  companion object {
    private const val TAG = "OpenWakeWord"
    private const val SAMPLE_RATE = 16000
    private const val FRAME_SAMPLES = 1280 // 80ms at 16kHz
    private const val MEL_FRAMES_PER_CHUNK = 5 // each 1280-sample chunk produces 5 mel frames
    private const val MEL_BANDS = 32
    private const val EMBEDDING_MEL_FRAMES = 76 // embedding model needs 76 mel frames
    private const val EMBEDDING_DIM = 96
    private const val WW_EMBEDDING_FRAMES = 16 // wake word model needs 16 embeddings
    private const val DEFAULT_THRESHOLD = 0.5f
    private const val DEBOUNCE_MS = 2000L // minimum ms between detections
  }

  private val running = AtomicBoolean(false)
  private var audioThread: Thread? = null
  private var audioRecord: AudioRecord? = null

  private var env: OrtEnvironment? = null
  private var melSession: OrtSession? = null
  private var embeddingSession: OrtSession? = null
  private var wakeWordSessions: MutableMap<String, Pair<OrtSession, String>> = mutableMapOf()

  // Ring buffers
  private val melBuffer = ArrayDeque<FloatArray>() // each entry is [32] mel band values
  private val embeddingBuffer = ArrayDeque<FloatArray>() // each entry is [96] embedding values

  var threshold: Float = DEFAULT_THRESHOLD
  var activeModels: List<String> = listOf("hey_jarvis_v0.1.onnx")
  private var lastDetectionMs = 0L

  /** Initialize ONNX sessions. Call on a background thread. */
  fun initialize(): Boolean {
    return try {
      env = OrtEnvironment.getEnvironment()
      val opts = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(2)
      }

      melSession = env!!.createSession(loadAsset("models/melspectrogram.onnx"), opts)
      embeddingSession = env!!.createSession(loadAsset("models/embedding_model.onnx"), opts)

      // Load available wake word models
      loadWakeWordModels(opts)

      Log.d(TAG, "Initialized: ${wakeWordSessions.size} wake word model(s) loaded")
      true
    } catch (e: Throwable) {
      Log.e(TAG, "Failed to initialize ONNX sessions", e)
      false
    }
  }

  private fun loadWakeWordModels(opts: OrtSession.SessionOptions) {
    wakeWordSessions.clear()
    val models = activeModels.ifEmpty { listOf("hey_jarvis_v0.1.onnx") }

    for (modelName in models) {
      try {
        val bytes = loadAsset("models/$modelName")
        val session = env!!.createSession(bytes, opts)
        // Get the input tensor name
        val inputName = session.inputNames.first()
        wakeWordSessions[modelName] = Pair(session, inputName)
        Log.d(TAG, "Loaded wake word model: $modelName (input=$inputName)")
      } catch (e: Throwable) {
        Log.w(TAG, "Failed to load wake word model: $modelName", e)
      }
    }
  }

  /** Start continuous listening. Does NOT steal audio focus. */
  fun start(): Boolean {
    if (running.get()) return true

    if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
      != PackageManager.PERMISSION_GRANTED
    ) {
      Log.w(TAG, "RECORD_AUDIO permission not granted")
      return false
    }

    if (wakeWordSessions.isEmpty()) {
      Log.w(TAG, "No wake word models loaded, cannot start")
      return false
    }

    val bufferSize = maxOf(
      AudioRecord.getMinBufferSize(
        SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
      ),
      FRAME_SAMPLES * 2 * 4, // at least 4 frames worth
    )

    try {
      audioRecord = AudioRecord(
        MediaRecorder.AudioSource.MIC,
        SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
        bufferSize,
      )

      if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
        Log.e(TAG, "AudioRecord failed to initialize")
        audioRecord?.release()
        audioRecord = null
        return false
      }
    } catch (e: Throwable) {
      Log.e(TAG, "AudioRecord creation failed", e)
      return false
    }

    running.set(true)
    melBuffer.clear()
    embeddingBuffer.clear()

    audioRecord?.startRecording()

    audioThread = thread(name = "OpenWakeWord", isDaemon = true) {
      val buffer = ShortArray(FRAME_SAMPLES)
      Log.d(TAG, "Audio capture thread started")

      while (running.get()) {
        val read = audioRecord?.read(buffer, 0, FRAME_SAMPLES) ?: -1
        if (read == FRAME_SAMPLES) {
          try {
            processAudioFrame(buffer)
          } catch (e: Throwable) {
            Log.e(TAG, "Error processing audio frame", e)
          }
        } else if (read < 0) {
          Log.w(TAG, "AudioRecord.read returned $read")
          break
        }
      }

      Log.d(TAG, "Audio capture thread ended")
    }

    Log.d(TAG, "Started listening")
    return true
  }

  /** Stop listening, release audio resources. */
  fun stop() {
    running.set(false)
    try {
      audioRecord?.stop()
    } catch (_: Throwable) { }
    try {
      audioRecord?.release()
    } catch (_: Throwable) { }
    audioRecord = null
    audioThread = null
    melBuffer.clear()
    embeddingBuffer.clear()
    Log.d(TAG, "Stopped listening")
  }

  /** Release ONNX sessions. */
  fun release() {
    stop()
    try {
      melSession?.close()
      embeddingSession?.close()
      wakeWordSessions.values.forEach { (session, _) -> session.close() }
    } catch (_: Throwable) { }
    melSession = null
    embeddingSession = null
    wakeWordSessions.clear()
    env = null
  }

  val isListening: Boolean get() = running.get()

  // ── Audio processing pipeline ──────────────────────────────────────

  private fun processAudioFrame(pcm16: ShortArray) {
    val environment = env ?: return
    val mel = melSession ?: return
    val emb = embeddingSession ?: return

    // 1. Convert PCM16 to float32 normalized [-1, 1]
    val audioFloat = FloatArray(pcm16.size) { pcm16[it].toFloat() / 32768f }

    // 2. Run melspectrogram model: input [1, 1280] → output [1, 1, 5, 32]
    val melInput = OnnxTensor.createTensor(
      environment,
      FloatBuffer.wrap(audioFloat),
      longArrayOf(1, audioFloat.size.toLong()),
    )
    val melResult = mel.run(mapOf("input" to melInput))
    val melOutput = melResult[0].value // shape: [1, 1, 5, 32]

    @Suppress("UNCHECKED_CAST")
    val melFrames = (melOutput as Array<Array<Array<FloatArray>>>)[0][0] // [5, 32]

    melInput.close()
    melResult.close()

    // 3. Accumulate mel frames
    for (frame in melFrames) {
      melBuffer.addLast(frame.copyOf())
      // Keep buffer bounded (slightly more than needed for embedding)
      while (melBuffer.size > EMBEDDING_MEL_FRAMES + MEL_FRAMES_PER_CHUNK) {
        melBuffer.removeFirst()
      }
    }

    // 4. When we have enough mel frames, compute embedding
    if (melBuffer.size < EMBEDDING_MEL_FRAMES) return

    // Take last 76 mel frames → [1, 76, 32, 1]
    val melSlice = melBuffer.toList().takeLast(EMBEDDING_MEL_FRAMES)
    val embInputData = FloatArray(EMBEDDING_MEL_FRAMES * MEL_BANDS)
    for (i in melSlice.indices) {
      System.arraycopy(melSlice[i], 0, embInputData, i * MEL_BANDS, MEL_BANDS)
    }

    val embInput = OnnxTensor.createTensor(
      environment,
      FloatBuffer.wrap(embInputData),
      longArrayOf(1, EMBEDDING_MEL_FRAMES.toLong(), MEL_BANDS.toLong(), 1),
    )
    val embResult = emb.run(mapOf("input_1" to embInput))

    @Suppress("UNCHECKED_CAST")
    val embVector = (embResult[0].value as Array<Array<Array<FloatArray>>>)[0][0][0] // [96]

    embInput.close()
    embResult.close()

    // 5. Accumulate embeddings
    embeddingBuffer.addLast(embVector.copyOf())
    while (embeddingBuffer.size > WW_EMBEDDING_FRAMES + 2) {
      embeddingBuffer.removeFirst()
    }

    // 6. When we have enough embeddings, classify
    if (embeddingBuffer.size < WW_EMBEDDING_FRAMES) return

    val embSlice = embeddingBuffer.toList().takeLast(WW_EMBEDDING_FRAMES)
    val wwInputData = FloatArray(WW_EMBEDDING_FRAMES * EMBEDDING_DIM)
    for (i in embSlice.indices) {
      System.arraycopy(embSlice[i], 0, wwInputData, i * EMBEDDING_DIM, EMBEDDING_DIM)
    }

    // 7. Run each wake word model
    for ((modelName, sessionPair) in wakeWordSessions) {
      val (session, inputName) = sessionPair
      val wwInput = OnnxTensor.createTensor(
        environment,
        FloatBuffer.wrap(wwInputData),
        longArrayOf(1, WW_EMBEDDING_FRAMES.toLong(), EMBEDDING_DIM.toLong()),
      )

      try {
        val wwResult = session.run(mapOf(inputName to wwInput))

        @Suppress("UNCHECKED_CAST")
        val score = (wwResult[0].value as Array<FloatArray>)[0][0]

        wwInput.close()
        wwResult.close()

        if (score > threshold) {
          val now = System.currentTimeMillis()
          if (now - lastDetectionMs > DEBOUNCE_MS) {
            lastDetectionMs = now
            Log.d(TAG, "Wake word detected! model=$modelName score=$score threshold=$threshold")
            onWakeWordDetected()
            // Clear embeddings to avoid repeated triggers
            embeddingBuffer.clear()
            return
          }
        }
      } catch (e: Throwable) {
        wwInput.close()
        Log.w(TAG, "Wake word inference failed: $modelName", e)
      }
    }
  }

  private fun loadAsset(path: String): ByteArray {
    return context.assets.open(path).use { it.readBytes() }
  }
}
