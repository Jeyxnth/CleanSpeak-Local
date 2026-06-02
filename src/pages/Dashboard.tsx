import { useCallback, useRef, useState } from "react";
import { useAuth } from "@/hooks/useAuth";
import UploadPanel from "@/components/dashboard/UploadPanel";
import PipelineVisualizer, { type StageData } from "@/components/dashboard/PipelineVisualizer";
import ResultsPanel from "@/components/dashboard/ResultsPanel";
import { useToast } from "@/hooks/use-toast";

const BACKEND_BASE_URL = (
  import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const initialStages = (): StageData[] =>
  Array.from({ length: 5 }, () => ({ status: "idle" as const }));

export default function Dashboard() {
  const { user } = useAuth();
  const { toast } = useToast();

  const [file, setFile] = useState<File | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [stages, setStages] = useState<StageData[]>(initialStages());

  const [pipelineRunning, setPipelineRunning] = useState(false);

  const [showResults, setShowResults] = useState(false);

  const [resultData, setResultData] = useState<any>(null);

  const [transcriptionMode, setTranscriptionMode] =
    useState<"normal" | "clear">("clear");

  const [applyNoiseReduction, setApplyNoiseReduction] =
    useState(true);

  // Draw waveform
  const drawWaveform = useCallback((audioFile: File) => {
    const canvas = canvasRef.current;

    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    if (!ctx) return;

    const reader = new FileReader();

    reader.onload = async (e) => {
      try {
        const audioCtx = new AudioContext();

        const buffer = await audioCtx.decodeAudioData(
          e.target?.result as ArrayBuffer
        );

        const data = buffer.getChannelData(0);

        const width = canvas.offsetWidth * window.devicePixelRatio;

        const height = canvas.offsetHeight * window.devicePixelRatio;

        canvas.width = width;

        canvas.height = height;

        ctx.clearRect(0, 0, width, height);

        const step = Math.ceil(data.length / width);

        const amp = height / 2;

        ctx.beginPath();

        ctx.strokeStyle = "hsl(263, 84%, 58%)";

        ctx.lineWidth = 1.5;

        for (let i = 0; i < width; i++) {
          let min = 1.0;

          let max = -1.0;

          for (let j = 0; j < step; j++) {
            const datum = data[i * step + j];

            if (datum < min) min = datum;

            if (datum > max) max = datum;
          }

          ctx.moveTo(i, (1 + min) * amp);

          ctx.lineTo(i, (1 + max) * amp);
        }

        ctx.stroke();

        audioCtx.close();

      } catch {

        // fallback waveform

        const width = canvas.offsetWidth * window.devicePixelRatio;

        const height = canvas.offsetHeight * window.devicePixelRatio;

        canvas.width = width;

        canvas.height = height;

        ctx.fillStyle = "hsl(263, 84%, 58%)";

        for (let i = 0; i < width; i += 4) {
          const h = Math.random() * height * 0.8;

          ctx.fillRect(i, (height - h) / 2, 2, h);
        }
      }
    };

    reader.readAsArrayBuffer(audioFile);

  }, []);

  const handleFileSelected = useCallback((f: File) => {

    setFile(f);

    setShowResults(false);

    setStages(initialStages());

    setTimeout(() => drawWaveform(f), 100);

  }, [drawWaveform]);

  const handleTranscriptionModeChange = useCallback(
    (mode: "normal" | "clear") => {
      setTranscriptionMode(mode);
    },
    []
  );

  const runPipeline = useCallback(async () => {

    if (!file || !user) return;

    setPipelineRunning(true);

    setShowResults(false);

    setStages(initialStages());

    const liveStages = initialStages();

    // Stage 1
    liveStages[0] = { status: "processing" };

    setStages([...liveStages]);

    try {

      const formData = new FormData();

      formData.append("file", file);

      formData.append("mode", transcriptionMode);

      formData.append(
        "apply_noise_reduction",
        String(applyNoiseReduction)
      );

      formData.append("language", "en");

      // Stage 1 complete
      liveStages[0] = {
        status: "complete",
        stats: {
          Upload: "Done",
        },
      };

      // Stage 2 processing
      liveStages[1] = {
        status: "processing",
      };

      setStages([...liveStages]);

      const response = await fetch(
        `${BACKEND_BASE_URL}/process`,
        {
          method: "POST",
          body: formData,
        }
      );

      const payload = await response.json();

      if (!response.ok) {

        const detail =
          typeof payload?.detail === "string"
            ? payload.detail
            : "Audio processing failed";

        throw new Error(detail);
      }

      // ===== FIXED RESPONSE MAPPING =====

      const rawTranscript =
        payload.transcript || "";

      const cleanedTranscript =
        payload.cleaned_transcript || rawTranscript;

      const fillersRemoved =
        payload.fillers_removed || 0;

      const wordsCensored =
        payload.words_censored || 0;

      const rawCount = rawTranscript.trim()
        ? rawTranscript.trim().split(/\s+/).length
        : 0;

      const cleanedCount = cleanedTranscript.trim()
        ? cleanedTranscript.trim().split(/\s+/).length
        : 0;

      const fillerWords: string[] = [];

      const censoredWords: string[] = [];

      const originalAudioUrl = undefined;

      const activeMode = transcriptionMode;

      // ===== COMPLETE PIPELINE =====

      const completedStages: StageData[] = [

        {
          status: "complete",
          stats: {
            Upload: "Done",
            Preprocessing: applyNoiseReduction
              ? "Enabled"
              : "Disabled",
          },
        },

        {
          status: "complete",
          stats: {
            "Words Detected": rawCount,
            Language: "English",
            Model: "Parakeet",
          },
        },

        {
          status: "complete",
          stats: {
            Cleanup: "Sarvam AI",
            Mode: activeMode,
          },
        },

        {
          status: "complete",
          stats: {
            "Fillers Removed": fillersRemoved,
            "Words Censored": wordsCensored,
          },
        },

        {
          status: "complete",
          stats: {
            Output: "Ready",
            "Cleaned Words": cleanedCount,
          },
        },
      ];

      setStages(completedStages);

      // ===== RESULTS =====

      const results = {
        rawTranscript,
        cleanedTranscript,
        fillerWords,
        censoredWords,
        originalAudioUrl,
        stats: {
          fillersRemoved,
          wordsCensored,
          originalWordCount: rawCount,
          cleanedWordCount: cleanedCount,
        },
      };

      setResultData(results);

      setShowResults(true);

    } catch (error: any) {

      liveStages[1] = {
        status: "error",
      };

      setStages([...liveStages]);

      toast({
        title: "Audio processing failed",
        description:
          error?.message ||
          "Could not generate cleaned audio",
        variant: "destructive",
      });

    } finally {

      setPipelineRunning(false);
    }

  }, [
    file,
    user,
    toast,
    transcriptionMode,
    applyNoiseReduction,
  ]);

  return (
    <div className="max-w-6xl mx-auto space-y-6 animate-fade-in">

      <div>
        <h1 className="text-3xl font-display font-bold">
          Dashboard
        </h1>

        <p className="text-text-secondary mt-1">
          Upload audio and run the CleanSpeech pipeline
        </p>
      </div>

      <UploadPanel
        onFileSelected={handleFileSelected}
        file={file}
        waveformCanvasRef={canvasRef}
        transcriptionMode={transcriptionMode}
        onTranscriptionModeChange={
          handleTranscriptionModeChange
        }
        applyNoiseReduction={applyNoiseReduction}
        onApplyNoiseReductionChange={
          setApplyNoiseReduction
        }
        onRunPipeline={runPipeline}
        pipelineRunning={pipelineRunning}
      />

      <PipelineVisualizer stages={stages} />

      {showResults && resultData && (
        <ResultsPanel {...resultData} />
      )}

    </div>
  );
}