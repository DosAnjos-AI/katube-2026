from pathlib import Path

# ============================================================================
# PROJECT ROOT DIRECTORY
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================================
# MODULE 00: NAMING - PIPELINE ENTRY POINT
# ============================================================================
# Runs ONCE per execution of main.py, before everything else. Scans
# `arquivos/input/` recursively, resolves an id for each audio file and
# MOVES the file to `arquivos/audios/{id}/{id}.<original format>`, which is
# the structure the rest of the pipeline consumes.
#
# ATTENTION - THE INPUT FOLDER IS EMPTIED: the file is MOVED, not copied.
# Always paste a COPY into `arquivos/input/`, never the only copy of the
# audio file.
NOMEACAO = {
    # How each audio file's id is decided
    # Options: "hash_md5" (RECOMMENDED) | "nome_original"
    #
    #   "hash_md5"      → the id is the MD5 of the file's CONTENT (bytes).
    #                     Trades away readability, and only the auxiliary
    #                     CSV (CSV_NOMEACAO) returns the origin of each
    #                     audio file. Three gains at once:
    #                       1. Two audio files with DIFFERENT content and
    #                          the same name generate different ids and
    #                          BOTH get in - new material is no longer
    #                          lost.
    #                       2. The SAME file pasted into another folder
    #                          generates the SAME id and is recognized as
    #                          a repeat - resuming after a break survives
    #                          until the source folder is renamed.
    #                       3. A safe id by construction: 32 hexadecimal
    #                          characters, with no space, accent or the
    #                          `|` that would break the dataset.csv row.
    #                     LIMITATION: the hash detects an IDENTICAL FILE,
    #                     not "the same sound content". The same audio
    #                     file re-exported, with different metadata or
    #                     converted to another format, generates a
    #                     different hash and passes as new.
    #
    #   "nome_original" → the id is the file name without the extension.
    #                     Readable, but collides between batches: two
    #                     batches with `entrevista.flac` compete for the
    #                     same id, and the second is blocked even if the
    #                     content is different - loss of material that
    #                     hash mode solves.
    #
    # The auxiliary CSV is written in BOTH modes: it is what fills the
    # `nome_original` column of dataset.csv.
    #
    # In BOTH modes the id is DETERMINISTIC - the same input always
    # produces the same id. This is mandatory: if the id varied between
    # runs, deduplication (CSV_CONCLUIDOS, below) would never block the
    # repeat and dataset.csv (pure append) would accumulate a duplicate
    # row.
    #
    # TIE-BREAK - only in "nome_original" mode. Identical names in
    # different folders get a numeric suffix: `entrevista`,
    # `entrevista_002`, `entrevista_003`, in the ALPHABETICAL order of
    # the relative path within `arquivos/input/`. In hash mode there is
    # NO tie-break: two files with the same name are already
    # distinguished by content, and if the content is the same they are
    # the same audio and should collide.
    'modo': 'nome_original',

    # Extensions accepted when scanning `arquivos/input/`
    # A file with an extension outside this list is REJECTED with a log
    # entry and a count, never silently ignored. Comparison always uses
    # `suffix.lower()`.
    #
    # This is the project's SINGLE SOURCE of input format: the
    # EXTENSOES_AUDIO below is derived from here.
    #
    # About `.opus`: FFmpeg 6.1.1 decodes it, and since m02 started
    # converting the input to WAV, it would cross the entire pipeline
    # without a problem - SoX, which cannot read opus, never sees the
    # original format again. It is OUT of the list by decision, not by
    # technical limitation. To enable it, just add '.opus' here.
    #
    # FFmpeg also decodes `aiff` and `au`, likewise outside the list.
    'formatos_entrada': {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'},
}


# ============================================================================
# NAMING AUXILIARY CSV - THE PROVENANCE OF EACH AUDIO FILE
# ============================================================================
# Holds the relationship `processed id` <-> `source path in arquivos/input/`.
# Written by M00 (pure append, one row per moved audio file) and READ by
# M14, which is what fills the `nome_original` column of dataset.csv.
#
# Path declared HERE, and not in the modules: two identical literals in
# different files diverge on the first edit, and M14 would start reading a
# file M00 does not write - with no error at all, just an empty column.
#
# Columns (`|` separator): nome_processado | nome_original | datetime_movido
CSV_NOMEACAO = PROJECT_ROOT / "dataset" / "nomeacao.csv"


# ============================================================================
# COMPLETED CSV - PERSISTENT DEDUPLICATION
# ============================================================================
# The list of audio files that FINISHED the pipeline. It is the ONLY
# source of deduplication: no audio file whose `nome_processado` is
# already listed here gets in again, whether in the same batch or in any
# batch at all.
#
# WHEN THE ROW IS WRITTEN: only after the COMPLETE success of the audio
# file - segments saved in `dataset/audio_dataset/{id}/`, rows committed
# to `dataset.csv` and the JSON backup copied to `historico_dataset/`. M14
# is the one that writes it, as the last step. An audio file that broke
# midway is NOT registered and can therefore be reprocessed - that is the
# reason the record comes last and not first.
#
# PURE APPEND. This file is only created and receives rows at the end.
# There is no code anywhere in the project that deletes a row or file
# from here. Reprocessing an already-completed audio file is a MANUAL
# user action: delete the corresponding row with an editor, outside the
# pipeline.
#
# THERE IS NO CONFIGURATION FIELD FOR "REPROCESS ANYWAY", and that is a
# decision, not an oversight. Since `dataset.csv` is pure append and
# nothing in it is removed, such a button would be a button for
# GENERATING A DUPLICATE in the dataset. Besides, in hash mode the
# criterion stopped being the name and became the content: "same name,
# different content" already gets in normally, which was the case that
# would justify the button.
#
# Path declared HERE, a single time, for the same reason as CSV_NOMEACAO:
# two identical literals in different files diverge on the first edit,
# and the guard would start reading a file nobody writes - with no error
# at all, just letting everything in again.
#
# Columns (`|` separator): nome_processado | nome_original | datetime_concluido
CSV_CONCLUIDOS = PROJECT_ROOT / "dataset" / "concluidos.csv"


# ============================================================================
# AUDIO FORMATS - SINGLE SOURCE
# ============================================================================
# Set of extensions that the pipeline modules recognize as audio.
# DERIVED from NOMEACAO['formatos_entrada'] - what comes in through the
# door is what the modules need to recognize, and two independent lists
# would diverge on the first edit. To change the accepted formats, edit
# the NOMEACAO block. Comparison always uses `suffix.lower()`.
#
# ----------------------------------------------------------------------------
# THE THREE FORMAT LISTS (confirmed empirically on the build installed on
# this machine - report 04, Ubuntu 24.04.4)
# ----------------------------------------------------------------------------
#
# 1) INPUT - what FFmpeg 6.1.1 can DECODE:
#       mp3, wav, flac, ogg, opus, m4a, aac, wma, aiff, au
#
#    The list IN EFFECT (NOMEACAO['formatos_entrada']) is more restrictive
#    by decision, not by limitation - see that field's comment.
#
# 2) INTERMEDIATE - `wav`, fixed and not configurable. m02 converts the
#    input to WAV when copying it to `temp/{id}/01-arquivos_originais/`,
#    preserving the original's sample rate, channels and bit depth. From
#    that point on the input format no longer circulates through the
#    pipeline - this is what allows accepting formats that SoX 14.4.2
#    cannot read (`m4a`, `aac`, `wma`, `opus`). m05 remains the only point
#    that produces 16 kHz.
#
# 3) OUTPUT - what SoX 14.4.2 can WRITE usefully:
#       flac, wav, ogg, mp3   (project default: flac - see SOX_NORMALIZER)
#
#    SoX also writes aiff, au, caf, w64, raw, voc and amr-nb, but these
#    are documented as NOT RECOMMENDED - `amr-nb`, in particular, forces
#    resampling to 8 kHz. Anyone needing another format converts outside
#    the project, with FFmpeg.
#
# ----------------------------------------------------------------------------
# ORIGIN OF THE BINARIES (report 04)
# ----------------------------------------------------------------------------
# `sox` v14.4.2 and `soxi` come from /usr/bin, i.e., from the OPERATING
# SYSTEM, NOT from the `katube-2026` conda env. The same applies to
# `ffmpeg` and `ffprobe` 6.1.1 (/usr/bin). Practical consequence:
# recreating the env from `environment.yml` does NOT reproduce the
# environment - the binaries need to be installed separately, on the
# system, and their version changes what the pipeline can read and write.
EXTENSOES_AUDIO = set(NOMEACAO['formatos_entrada'])


# ============================================================================
# MASTER BLOCK - Active Module Control
# ============================================================================
# Turns the system's main modules on/off
# True = module will run | False = module will be skipped
MASTER = {
    # Options: 'vad'. The value '' (audio already segmented) is provided for
    # in the code but DOES NOT WORK: no module feeds 02-segmentos_originais
    # without m04, and m05 aborts the pipeline in 0.00 min for lack of the
    # JSON. Do not use it until an entry path for ready-made segments
    # exists.
    'segmentacao': 'vad',
    'mos_filter': True,        # True = use it; False = do not use it
    'overlap': True,           # Whether to use the overlap detector or not
    'transcricao_whisper': True,
    'transcricao_wav2vec': True,
    'Denoiser': True,
    'cleanup': 'all',          # Options: 'all' (input+temp), 'input' (input only), 'temp' (temp only), 'none' (does not delete)

    # ------------------------------------------------------------------------
    # CPU FALLBACK - single, global control for all models
    # ------------------------------------------------------------------------
    # Governs what happens when a block requests 'gpu' and the GPU does not
    # deliver (CUDA absent, CUDA error, VRAM shortage, any exception while
    # loading the model).
    #
    #   True  -> the pipeline FALLS BACK TO CPU and continues. The original
    #            exception and the device actually used are logged at ERROR
    #            level, on stderr, with the [ERRO] prefix. The loud log is
    #            mandatory: a run that falls back to CPU without anyone
    #            noticing costs about 9.7x real time and passes as normal.
    #
    #   False -> NO fallback. The error brings the module down, which
    #            returns False per the return contract, and main.py marks
    #            the audio file as failed in processamento_metadados.csv.
    #
    # Applies to the five models (Whisper, wav2vec, pyannote, SQUIM and
    # DeepFilterNet3). See the DeepFilterNet3 CAVEAT in
    # DEEPFILTERNET_DENOISER['device'].
    'fallback_cpu': True,
}


# =============================================================================
# MODULE 01: VAD AUDIO SEGMENTER (VOICE ACTIVITY DETECTION)
# =============================================================================

# Configuration for the automatic segmentation module using voice detection
# Used when MASTER['segmentacao'] = 'vad'
#
# This module uses Silero-VAD to automatically detect moments of speech
# and silence in the audio, creating segments based on natural pauses.
#
# IMPORTANT: Always processes audio at 16 kHz internally (automatic conversion)
SEGMENTADOR_AUDIO_VAD = {

    # ------------------------------------------------------------------------
    # Voice Detection (Voice Activity Detection)
    # ------------------------------------------------------------------------
    'deteccao': {
        # Confidence threshold to consider that voice is present (0.0 to 1.0)
        # - LOW values (0.3-0.4): more sensitive, detects even whispers/noise
        # - MEDIUM values (0.5): balanced, recommended for most cases
        # - HIGH values (0.6-0.8): less sensitive, only detects clear voice
        # Example: 0.5 = medium confidence (50%) to consider it as voice
        'voice_threshold': 0.5,

        # Size of the analysis window, in seconds
        # Defines the temporal granularity of voice detection
        # - Smaller values (0.05-0.1): higher precision, more processing
        # - Medium values (0.15): recommended balance
        # - Larger values (0.3-0.5): less precision, faster
        # Example: 0.15 = analyzes the audio in 150ms blocks
        'window_size_seconds': 0.15,
    },

    # ------------------------------------------------------------------------
    # Silence and Speech Criteria
    # ------------------------------------------------------------------------
    'criterios': {
        # Minimum SPEECH duration to be considered valid (milliseconds)
        # Shorter speech is ignored (considered noise)
        # - Low values (100-200ms): captures very brief speech
        # - Medium values (250-500ms): filters short noise (recommended)
        # - High values (>500ms): only accepts long speech
        # Example: 250 = ignores sounds shorter than 250ms (0.25s)
        'min_speech_duration_ms': 250,

        # Minimum SILENCE duration between speech (milliseconds)
        # Shorter silences do not separate speech (stay in the same segment)
        # - Low values (50-100ms): splits on very brief pauses
        # - Medium values (100-200ms): balance (recommended)
        # - High values (>300ms): only splits on long pauses
        # Example: 100 = pauses shorter than 100ms do not split the segment
        'min_silence_duration_ms': 100,

        # Minimum SILENCE duration to FORCE a segment split (seconds)
        # Pauses longer than this value always create a new segment
        # - Low values (0.2-0.3s): sensitive to short pauses
        # - Medium values (0.3-0.5s): balance (recommended)
        # - High values (>0.5s): only splits on very long pauses
        # Example: 0.3 = a 300ms pause always creates a new segment
        'min_silence_for_split': 0.3,
    },

    # ------------------------------------------------------------------------
    # Padding (Safety Margin on Cuts)
    # ------------------------------------------------------------------------
    'padding': {
        # Extra time at the START of each segment (milliseconds)
        # Avoids cutting off the start of the first word
        # - Low values (10-30ms): more precise cut
        # - Medium values (30-50ms): recommended safety margin
        # - High values (>100ms): may include extra silence
        #
        # ATTENTION - this value is applied TWICE at the start: once by
        # Silero (m04 passes it as speech_pad_ms, and Silero expands both
        # sides) and again by m04 itself when assembling the group. With
        # 30 ms configured, the start receives 60 ms in total. The end
        # receives inicio_ms (via Silero) + fim_ms (via m04).
        'inicio_ms': 30,

        # Extra time at the END of each segment (milliseconds)
        # Avoids cutting off the end of the last word
        # - Low values (10-30ms): more precise cut
        # - Medium values (30-50ms): recommended safety margin
        # - High values (>100ms): may include extra silence
        # Example: 30 = adds 30ms after the detected end of speech
        # This field does NOT reach Silero: only m04 applies it.
        'fim_ms': 30,
    },

    # ------------------------------------------------------------------------
    # Duration Limits of the Final Segments
    # ------------------------------------------------------------------------
    'segmentos': {
        # TARGET minimum duration of each segment, in seconds
        # Shorter segments are grouped with the next ones
        # - Low values (2-4s): accepts very short segments
        # - Medium values (4-8s): balance (recommended)
        # - High values (>10s): forces long segments
        #
        # ATTENTION: the EFFECTIVE minimum is (min_seg - tolerancia), not
        # min_seg. With min_seg=4.0 and tolerancia=0.5, the real minimum
        # is 3.5 s.
        'min_seg': 4.0,

        # GROUPING ceiling, in seconds - NOT an absolute ceiling
        # Prevents adding one more speech chunk to a group that has
        # already reached this limit. A single speech chunk LARGER than
        # the limit CROSSES IT WHOLE - nothing splits it.
        # - Low values (8-12s): forces short segments
        # - Medium values (15-20s): balance (recommended)
        # - High values (>25s): allows very long segments
        #
        # Measured in this project: with max_seg=8.0, one segment came
        # out at 8.312 s. With 15.0, segments ranged between 6.5 and
        # 12.3 s.
        'max_seg': 15.0,

        # Tolerance on durations (seconds)
        # Allows small variations on the min/max limits
        # - Low values (0.3-0.5s): stricter
        # - Medium values (0.8-1.0s): balance (recommended)
        # - High values (>1.5s): more flexible
        # Example with the current values: 0.5 accepts a 3.5 s segment
        # (min_seg 4.0 - tolerancia 0.5)
        'tolerancia': 0.5,
    },

    # ------------------------------------------------------------------------
    # General Behavior
    # ------------------------------------------------------------------------
    'comportamento': {
        # FIELD WITH NO EFFECT TODAY.
        # The skip code exists (m04), but m02 erases the entire
        # temp/{audio_id} at the start of every processing run: by the
        # time the check happens, the destination folder was just created
        # empty, and the answer is always "there is nothing to skip".
        # Confirmed over seven runs reprocessing the same audio_id - the
        # skip message never appeared. Turning this on requires first
        # deciding which invariant wins: the clean restart or reuse.
        'sobrescrever': False,
    },
}

# =============================================================================
# MODULE 02: MOS FILTER (MEAN OPINION SCORE - AUDIO QUALITY)
# =============================================================================

# Configuration for the MOS-based audio quality filter
# Evaluates audio using the SQUIM model (Speech Quality and Intelligibility Measures)
# Classifies segments as low, medium or high quality
MOS_FILTER = {

    # ------------------------------------------------------------------------
    # Processing Device
    # ------------------------------------------------------------------------
    # Defines where the MOS model will run
    # Available options - ONLY TWO ("auto" was eliminated):
    # - "gpu": requests GPU/CUDA. If the GPU does not deliver, what happens
    #          is decided by MASTER['fallback_cpu'], not by this field.
    # - "cpu": forces CPU use (slower, but works on any machine)
    #
    # Any other value is REJECTED with an explicit error in m01 - there is
    # no silent resolution or hidden default value.
    # Note: GPU speeds things up significantly (3-5x faster)
    'device': 'gpu',

    # ------------------------------------------------------------------------
    # Quality Thresholds (MOS Score)
    # ------------------------------------------------------------------------
    # MOS (Mean Opinion Score) ranges from 1.0 (terrible) to 5.0 (excellent)
    # Defines the thresholds for quality classification

    'thresholds': {
        # Minimum acceptable threshold
        # Audio files with MOS < min_threshold are DISCARDED
        # Typical values: 1.5-2.5
        # Example: 2.0 = discards very bad audio files
        'min_threshold': 2.0,

        # Threshold for high quality
        # Audio files with MOS >= max_threshold are classified as 'alta'
        # Typical values: 3.0-4.0
        # Example with the current value: MOS >= 3.0 is classified as 'alta'
        #
        # This field does NOT decide who goes through the denoiser - that
        # is decided by DEEPFILTERNET_DENOISER['mos_quality_filter'].
        'max_threshold': 3.0,

        # Intermediate range (calculated automatically):
        # min_threshold <= MOS < max_threshold
        # These audio files go through denoising before the final dataset
    },

    # ------------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------------
    # Processes multiple audio files simultaneously for greater efficiency

    'batch': {
        # Batch size (how many audio files to process together)
        # Larger values = faster, but uses more VRAM
        #
        # Options:
        # - "auto": in this block it does NOT look at VRAM. m06 returns 8
        #           for GPU and 1 for CPU, and that's it. (STT_WHISPER's
        #           "auto" is different: there VRAM is actually read.)
        # - 1-16: Fixed value (larger numbers require more VRAM)
        #
        # VRAM usage reference (approximate):
        # - batch_size=1:  ~2.0 GB
        # - batch_size=4:  ~3.0 GB
        # - batch_size=8:  ~4.0 GB
        # - batch_size=16: ~6.0 GB
        #
        # Recommendation:
        # - GPU with 24GB: "auto" or 16
        # - GPU with 8-16GB: 8
        # - GPU with 4-8GB: 4
        # - CPU: 1-2
        'batch_size': 'auto',
    },

    # ------------------------------------------------------------------------
    # General Behavior
    # ------------------------------------------------------------------------
    'comportamento': {
        # FIELD WITH NO EFFECT TODAY.
        # Same case as the VAD block: the skip code exists (m06), but m02
        # erases temp/{audio_id} beforehand, so the JSON the check looks
        # for is never there. Turning this on is a project decision, not
        # an implementation one.
        'sobrescrever': False,
    },
}

# =============================================================================
# MODULE 04: OVERLAP DETECTOR 01 (SPEAKER OVERLAP)
# =============================================================================

# Configuration for the overlap detector using speaker diarization
# Detects whether there is speech overlap (multiple speakers talking at once)
# Uses the pyannote model for audio analysis
OVERLAP_DETECTOR = {

    # ------------------------------------------------------------------------
    # Processing Device
    # ------------------------------------------------------------------------
    # Defines where the model will run
    # Available options - ONLY TWO ("auto" was eliminated):
    # - "gpu": requests GPU/CUDA. If the GPU does not deliver, what happens
    #          is decided by MASTER['fallback_cpu'], not by this field.
    # - "cpu": forces CPU use (slower, but works on any machine)
    #
    # Any other value is REJECTED with an explicit error in m01.
    # Note: GPU speeds up processing significantly
    'device': 'gpu',

    # ------------------------------------------------------------------------
    # Diarization Model
    # ------------------------------------------------------------------------
    # HuggingFace model for overlap detection.
    # This field IS read by m01 and actually defines the model loaded -
    # previously there was a constant in the code that ignored it.
    # IMPORTANT: Requires a HuggingFace token configured in .env
    'modelo': 'pyannote/speaker-diarization-3.1',

    # ------------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------------
    'batch': {
        # ONLY SUPPORTED VALUE: 1
        #
        # This module processes one segment at a time, with an individual
        # timeout per audio file - there is no batch path in m07. The
        # field IS READ, and any value other than 1 is REJECTED with an
        # explicit error, which brings the module down. Previously the
        # value was simply ignored, and anyone who configured 8 had no
        # way of noticing.
        #
        # Batch processing here would be new functionality, not a
        # configuration adjustment.
        'batch_size': 1,
    },
    'timeout': {
    'por_audio_segundos': 150,  # Maximum timeout per audio file
    },

    # ------------------------------------------------------------------------
    # General Behavior
    # ------------------------------------------------------------------------
    'comportamento': {
        # FIELD WITH NO EFFECT TODAY - and with no reader at all.
        # Nothing in m07 checks the overlap01 field before processing;
        # this value is not consulted by any line of the project.
        'sobrescrever': False,
    },
}

# =============================================================================
# MODULE 05: STT WHISPER (SPEECH-TO-TEXT)
# =============================================================================

# Configuration for the transcription module using Whisper
# Converts audio segments into text using the distil-whisper PT-BR model
# Model: freds0/distil-whisper-large-v3-ptbr
STT_WHISPER = {

    # ------------------------------------------------------------------------
    # Processing Device
    # ------------------------------------------------------------------------
    # Defines where the Whisper model will run
    # Available options - ONLY TWO ("auto" was eliminated):
    # - "gpu": requests GPU/CUDA. If the GPU does not deliver, what happens
    #          is decided by MASTER['fallback_cpu'], not by this field.
    # - "cpu": forces CPU use (slower, but works on any machine)
    #
    # Any other value is REJECTED with an explicit error in m01.
    # Note: GPU speeds things up significantly (6x faster than large-v3)
    'device': 'gpu',

    # ------------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------------
    # Processes multiple audio files simultaneously for greater efficiency

    'batch': {
        # Batch size (how many audio files to transcribe together)
        # Larger values = faster, but uses more VRAM
        #
        # Options:
        # - "auto": Calculates automatically based on available VRAM
        # - 1-16: Fixed value (larger numbers require more VRAM)
        #
        # VRAM usage reference (approximate):
        # - batch_size=1:  ~2.5 GB
        # - batch_size=4:  ~4.0 GB
        # - batch_size=8:  ~6.0 GB
        # - batch_size=16: ~10.0 GB
        #
        # Recommendation:
        # - GPU with 24GB: "auto" or 16
        # - GPU with 8-16GB: 8
        # - GPU with 4-8GB: 4
        # - CPU: automatic 1
        'batch_size': 8,
    },

    # ------------------------------------------------------------------------
    # General Behavior
    # ------------------------------------------------------------------------
    'comportamento': {
        # FIELD WITH NO EFFECT TODAY - and with no reader at all.
        # Nothing in m08 checks the stt_whisper field before
        # transcribing; this value is not consulted by any line of the
        # project.
        'sobrescrever': False,
    },
}

# ============================================================
# MODULE 06: STT WAV2VEC2 (SPEECH-TO-TEXT)
# ============================================================
STT_WAV2VEC2= {
    # Processing device - ONLY "gpu" or "cpu" ("auto" was eliminated).
    # With "gpu", the fallback to CPU is decided by
    # MASTER['fallback_cpu']. Any other value is rejected with an error
    # in m01.
    "device": "gpu",
}


# ============================================================
# MODULE 07: TEXT NORMALIZER
# ============================================================
TEXT_NORMALIZER = {

    # Punctuation that affects the speaker's diction/pronunciation
    # Removes: . , ; ! ? _
    # Example: "Olá, mundo!" → "Olá mundo" (with remove=True)
    # Example: "Olá, mundo!" → "Olá, mundo!" (with remove=False)
    # Useful for: STT comparison where punctuation is not transcribed
    "remove_punctuation_diction": True,

    # Graphic accentuation (diacritical marks)
    # Removes EVERY diacritical mark, via Unicode NFD decomposition
    # followed by discarding the Mn category: acute, grave, circumflex,
    # tilde AND CEDILLA.
    # Verified examples: "josé" → "jose", "coração" → "coracao",
    #                    "açúcar" → "acucar" (the cedilla goes too)
    # Example: "josé" → "josé" (with remove=False)
    # IMPORTANT: If False, spelled-out numbers will have accents (três, décimo)
    #            If True, spelled-out numbers without accents (tres, decimo)
    # Useful for: normalization for models that do not handle accents well
    "remove_accents_graphic": True,
}

# ============================================================
# MODULE 08: SIMILARITY VALIDATOR
# ============================================================
SIMILARITY_VALIDATOR = {

    # All THREE metrics are ALWAYS calculated, comparing the Whisper and
    # Wav2Vec2 transcriptions against each other. There is no choice of
    # metric: the segment only continues in the pipeline if it passes all
    # THREE thresholds below.
    #
    # ATTENTION TO THE DIRECTION: each metric uses its own established
    # convention, and they do NOT point the same way. In WER and CER,
    # which are ERROR rates, lower is better. In levenshtein_norm, which
    # is a SIMILARITY, higher is better. That is why there are three
    # fields, and not an average: they measure different things, in
    # different directions.
    #
    # Raising an error threshold (wer, cer) LOOSENS the filter.
    # Raising the similarity threshold (levenshtein_norm) TIGHTENS the
    # filter.

    # ------------------------------------------------------------------------
    # WER - Word Error Rate (error rate per WORD)
    # ------------------------------------------------------------------------
    # Measures: WORD edits needed to transform one transcription into the
    #       other, divided by the number of words in the reference.
    # Direction: 0.0 = identical transcriptions. THE LOWER, THE BETTER.
    # Passes if: wer <= threshold
    # NO UPPER BOUND: goes above 1.0 when one of the models inserts more
    #       words than the reference has (e.g., a 1-word reference and a
    #       5-word hypothesis gives WER 4.0).
    # Example: "casa azul" vs "casa verde" → 1 word swapped out of 2 → 0.50
    # This threshold is the most LENIENT on purpose: a single wrong letter
    # invalidates the whole word, so WER is always harsher than CER over
    # the same text.
    # Typical values: 0.30 (permissive), 0.20 (balanced), 0.10 (strict)
    #
    # CALIBRATED ON REAL DATA (task 28), not at the desk. wav2vec is
    # PHONETIC and does not correct for Portuguese the way Whisper does:
    # WORD divergence between the two is expected and is NOT a sign of
    # bad audio. With CER and levenshtein_norm already filtering, WER is
    # the loosest of the three.
    #
    # Measurement of the 23 eligible segments from stage 6 (report 27):
    # eleven failed ONLY on WER, with CER between 0.0642 and 0.1481
    # (threshold 0.15) and levenshtein_norm between 0.8519 and 0.9358
    # (threshold 0.85) - i.e., comfortable on the two metrics that
    # actually measure transcription error. The value 0.20 failed 100% of
    # the segments of two out of the five audio files, while the SAME
    # material in another cut passed with 0.11.
    #
    # Why 0.35 and not 0.40: at 0.40, WER would stop deciding on its own
    # in any case in the sample - everything it would block is already
    # blocked by CER and levenshtein_norm - and it would become a button
    # with no effect. At 0.35 it still blocks one segment on its own (WER
    # 0.4000 with CER 0.1481 and levenshtein_norm 0.8519, scraping by on
    # all three), so it keeps its own decision-making power as a net
    # against gross structural divergence.
    "limiar_wer": 0.35,

    # ------------------------------------------------------------------------
    # CER - Character Error Rate (error rate per CHARACTER)
    # ------------------------------------------------------------------------
    # Measures: CHARACTER edits needed to transform one transcription into
    #       the other, divided by the number of characters in the
    #       reference.
    # Direction: 0.0 = identical transcriptions. THE LOWER, THE BETTER.
    # Passes if: cer <= threshold
    # Also has no upper bound, for the same reason as WER.
    # More granular than WER: catches a subtle error (accent, swapped
    # letter) without condemning the whole word.
    # Example: "casa azul" vs "casa verde" → 5 edits in 9 chars → 0.5556
    # Typical values: 0.25 (permissive), 0.15 (balanced), 0.08 (strict)
    "limiar_cer": 0.15,

    # ------------------------------------------------------------------------
    # NORMALIZED LEVENSHTEIN (CHARACTER similarity)
    # ------------------------------------------------------------------------
    # Measures: 1 - (Levenshtein distance / max length of the two texts)
    # Direction: 1.0 = identical texts. THE HIGHER, THE BETTER. (INVERSE of
    #       the two metrics above - careful when adjusting.)
    # Passes if: levenshtein_norm >= threshold
    # Closed range between 0.0 and 1.0, unlike WER and CER.
    # Example: "gato" vs "pato" → 1 edit in 4 chars → 0.75
    # Typical values: 0.75 (permissive), 0.85 (balanced), 0.92 (strict)
    "limiar_levenshtein_norm": 0.85,
}

# ============================================================
# MODULE 09: DEEPFILTERNET DENOISER
# ============================================================
DEEPFILTERNET_DENOISER = {

    # MOS quality filter for selecting audio files to process
    # Array with the desired quality categories
    # Available options: "alta", "media", "baixa"
    # Usage examples:
    #   ["alta", "media", "baixa"] → Processes ALL audio files (regardless of MOS)
    #   ["alta"] → Processes ONLY high-quality audio files
    #   ["media", "baixa"] → Processes medium and low-quality audio files
    #   [] → PROCESSES NO AUDIO FILE (empty array = no processing)
    #
    # ATTENTION - what actually reaches this point:
    #   "baixa" NEVER arrives: m06 discards those segments earlier, via
    #   min_threshold. Only "media" and "alta" remain.
    #
    #   HOW MUCH ["media"] SELECTS, measured in task 28 (5 audio files, 23
    #   eligible segments): 3 segments, all from the `entrevista` audio
    #   file. The other 14 delivered did not go through the denoiser. In
    #   other words, the filter actually selects, and BOTH m13 paths
    #   (with and without denoiser) are exercised in the same run.
    #
    #   The previous comment stated that ["media"] selected ZERO segments
    #   "in every test run". That held when the test set was only
    #   `qMrMmG__Yhw_60s`, whose segments come out almost all as "alta"
    #   (MOS from 3.36 to 3.89 with max_threshold=3.0). It stopped
    #   holding once audio files from another source came in.
    #   To process EVERY segment with the denoiser, include "alta".
    #
    # IMPORTANT: The original (input) files ALWAYS remain intact
    #            Denoising creates new processed files in the output
    "mos_quality_filter": ["media"],

    # Processing device - ONLY TWO options ("auto" was eliminated):
    #   "gpu" → requests GPU/CUDA; the fallback to CPU is decided by
    #           MASTER['fallback_cpu'], not by this field
    #   "cpu" → forces CPU processing (slower, always available)
    # Any other value is REJECTED with an explicit error in m01.
    #
    # DEEPFILTERNET3 CAVEAT (finding A29, measured on the installed
    # build): the library resolves the device on its own, with
    # get_device(), which picks cuda:0 whenever CUDA is available.
    # init_df() does NOT accept a device parameter, and the model's
    # internal config.ini has [train] device = (empty). To force the
    # library onto this field, m01 writes the DEVICE key into
    # DeepFilterNet's internal config right AFTER init_df() - doing it
    # before is useless, because init_df() itself reloads the parser and
    # erases the adjustment.
    #
    # KNOWN AND ACCEPTED LIMITATION (Mestre's decision, instruction 20):
    # init_df() itself keeps loading the model on the device the library
    # chooses. Consequence: on a GPU that FAILS, DeepFilterNet3 CANNOT
    # restart on CPU, because the retry's init_df() goes back to trying
    # CUDA. The other four models have full fallback. Fixing this would
    # require a copy of the model inside the project with config.ini
    # edited - a cost not justified today.
    "device": "gpu",

    # Turns the DeepFilterNet post-filter on/off.
    # The library's parameter is BOOLEAN (signature verified on the
    # installed build, DeepFilterNet 0.5.6: post_filter: bool = False):
    #   0 → turns off the post-filter
    #   any other value → turns it on
    #
    # There are NOT three degrees of aggressiveness in this version: the
    # value 2 is identical to 1. The post-filter does an extra, smaller
    # noise reduction.
    "post_filter": 1,

    # Noise attenuation limit, in DECIBELS - NOT a fraction from 0 to 1.
    # The value travels to enhance()'s `atten_lim_db` parameter, which
    # mixes the original and treated signal like this:
    #
    #     lim = 10^(-|dB|/20)
    #     output = original * lim + treated * (1 - lim)
    #
    # In other words, the fraction of denoising that reaches the audio is
    # (1 - lim), and a HIGHER decibel value means MORE denoising - the
    # scale is the inverse of what the old comment described.
    #
    # MEASURED IN THIS PROJECT (task 19, 6 values x 6 segments):
    #    0.95 dB -> applies  10.4 % of the denoising (practically inert)
    #    3.0  dB -> applies  29.2 %
    #    6.0  dB -> applies  49.9 %
    #   12.0  dB -> applies  74.9 %
    #   24.0  dB -> applies  93.7 %
    #   40.0  dB -> applies  99.0 %
    #
    # BELOW 6 dB: inert region - the module loads, runs and barely
    # changes the audio (at 0.95 dB the measured difference was 51 dB
    # below the signal, inaudible).
    # ABOVE 24 dB: saturation - from 24 to 40 dB the gain is 0.5 dB.
    # Equivalent to having no limit at all.
    # 0 or None: no limit, full denoising.
    #
    # USEFUL RANGE: 6 to 24 dB. Working value: 12.0 (75 % of the
    # denoising, conservative enough not to mistreat the voice in a TTS
    # dataset). Audio duration does NOT change at any tested value - the
    # duration invariant is not affected by this field.
    "attenuation_limit": 12.0,

    # Skip segments already processed before
    # Checks the "utilizou_denoiser" field in the dynamic JSON
    #   True → Ignores audio files with an existing flag (saves processing)
    #   False → Reprocesses all eligible audio files (overwrites outputs)
    #
    # Use case True: Re-runs after failures/interruptions
    # Use case False: Parameter change (post_filter, attenuation_limit)
    #
    # IMPORTANT: The flag only prevents reprocessing, it does not validate quality
    "skip_if_already_processed": True,
}

# ============================================================
# MODULE 10: SOX AUDIO NORMALIZER
# ============================================================
SOX_NORMALIZER = {

    # Sample rate of the output audio
    # Common values: 8000, 16000, 22050, 44100, 48000 (in Hz)
    # Usage examples by application:
    #   8000  → Telephony (minimum quality)
    #   16000 → STT (Speech-to-Text) - quality/performance balance
    #   22050 → TTS (Text-to-Speech) - intermediate quality
    #   44100 → CD quality - general high-fidelity use
    #   48000 → Professional audio/broadcasting
    #
    # Current value: 24000 Hz, project's choice for TTS. It is on the
    # list of common values and does not contradict anything - STT/TTS
    # models usually expect 16000, 22050 or 24000 Hz.
    # Higher values = higher quality but higher computational cost
    "sample_rate": 24000,

    # Bit depth of the audio
    # Values: 16, 24, 32 (in bits)
    #   16 → Standard for STT/TTS (enough for speech)
    #   24 → Greater dynamic range (professional use)
    #   32 → Maximum quality (rarely needed for AI)
    #
    # Recommendation: 16 bits for AI training datasets
    # Higher values increase size with no significant gain
    "bit_depth": 16,

    # Number of audio channels
    # Values: 1 (mono), 2 (stereo)
    #   1 → REQUIRED for STT/TTS (models expect mono)
    #   2 → Only if preserving spatialization is necessary
    #
    # ATTENTION - CRITICAL: Practically all STT/TTS models require MONO
    # Stereo doubles the size with no benefit for training
    "channels": 1,

    # Output audio file format
    # Options: "wav", "flac", "mp3", "ogg"
    #   "wav"  → No compression, maximum quality, large files
    #   "flac" → Lossless compression, quality=WAV, ~50% smaller
    #   "mp3"  → Lossy compression, OK quality, small files
    #   "ogg"  → Lossy compression, better than MP3, less compatible
    #
    # Recommendation by case:
    #   AI training → "flac" (perfect quality, storage savings)
    #   Deployment → "mp3" (lower loading latency)
    #   Archiving → "wav" (lossless, full compatibility)
    "output_format": "flac",

    # Volume normalization method
    # Options: "peak", "rms", "loudness" - what EACH ONE emits to the SoX
    # 14.4.2 installed on this machine:
    #   "peak"     → `norm <dB>`: normalizes by PEAK
    #   "rms"      → `gain -n <dB>`: ALSO normalizes by PEAK, not by
    #                average energy. The installed build's
    #                `sox --help-effect=gain` documents `-n` as "Norm
    #                file to 0dBfs". True RMS in SoX would be `gain -b` /
    #                `-B`.
    #   "loudness" → `loudness <dB>`: SoX's ISO 226 audibility curve.
    #                It is NOT LUFS/EBU R128.
    #
    # Practical consequence: "peak" and "rms" deliver the same thing -
    # peak normalization, with minor implementation differences. The
    # name "rms" does not describe what happens.
    "normalize_method": "rms",

    # Target normalization level, in decibels
    # Typical values: -3.0, -1.0, 0.0 (in dB)
    #   -3.0 → Conservative (headroom to avoid clipping)
    #   -1.0 → Balanced (broadcasting standard)
    #   0.0  → Maximum (no safety margin)
    #
    # ATTENTION: Positive values cause distortion (clipping)
    # Very negative values result in quiet audio
    # Recommendation: -3.0 for training datasets
    "target_level_db": -3.0,
}