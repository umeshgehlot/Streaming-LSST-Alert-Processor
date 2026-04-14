import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import api from "./api";

const modelOptions = ["autoencoder", "vae", "transformer"];
const MAX_CHART_POINTS = 1200;
const dashboardTabs = [
  { id: "pipeline", label: "Pipeline" },
  { id: "charts", label: "Charts" },
  { id: "comparison", label: "Comparison" },
  { id: "history", label: "History" },
  { id: "innovation", label: "Innovation" },
  { id: "agent", label: "Agent" },
];

function downsampleByIndex(data, maxPoints) {
  if (data.length <= maxPoints) {
    return data;
  }
  const step = Math.ceil(data.length / maxPoints);
  const sampled = [];
  for (let index = 0; index < data.length; index += step) {
    sampled.push(data[index]);
  }
  if (sampled[sampled.length - 1] !== data[data.length - 1]) {
    sampled.push(data[data.length - 1]);
  }
  return sampled;
}

function App() {
  const [screen, setScreen] = useState("home");
  const [isAuthenticated, setIsAuthenticated] = useState(Boolean(localStorage.getItem("astro_token")));
  const [userProfile, setUserProfile] = useState(null);
  const [activeTab, setActiveTab] = useState("pipeline");
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [datasetId, setDatasetId] = useState("");
  const [filename, setFilename] = useState("");
  const [pointsCount, setPointsCount] = useState(0);
  const [normalizedPoints, setNormalizedPoints] = useState([]);
  const [selectedModels, setSelectedModels] = useState(["autoencoder", "vae", "transformer"]);
  const [detectModel, setDetectModel] = useState("autoencoder");
  const [recentOnly, setRecentOnly] = useState(true);
  const [recentYears, setRecentYears] = useState(2);
  const [thresholdPercentile, setThresholdPercentile] = useState(95);
  const [trainingSummary, setTrainingSummary] = useState([]);
  const [comparisonSummary, setComparisonSummary] = useState([]);
  const [bestModel, setBestModel] = useState("");
  const [datasetMeta, setDatasetMeta] = useState(null);
  const [datasetStats, setDatasetStats] = useState(null);
  const [scores, setScores] = useState([]);
  const [xaiHeatmap, setXaiHeatmap] = useState([]);
  const [ensembleConfidence, setEnsembleConfidence] = useState([]);
  const [anomalyIndices, setAnomalyIndices] = useState([]);
  const [resultId, setResultId] = useState("");
  const [historyItems, setHistoryItems] = useState([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyModelFilter, setHistoryModelFilter] = useState("all");
  const [historyOffset, setHistoryOffset] = useState(0);
  const [historyLimit] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [jobInfo, setJobInfo] = useState(null);
  const [capabilities, setCapabilities] = useState({ vertical: [], horizontal: [] });
  const [liveEvents, setLiveEvents] = useState([]);
  const [agentOutput, setAgentOutput] = useState(null);
  const [agentActivities, setAgentActivities] = useState([]);
  const [reasoningText, setReasoningText] = useState("");
  const [reasoningCursor, setReasoningCursor] = useState(0);
  const [discoveries, setDiscoveries] = useState([]);
  const [discoverySummary, setDiscoverySummary] = useState(null);
  const [rlStats, setRlStats] = useState(null);
  const [insightSummary, setInsightSummary] = useState("");
  const [insightBackend, setInsightBackend] = useState("");
  const [latentPoints, setLatentPoints] = useState([]);
  const [latentMethod, setLatentMethod] = useState("umap-style");
  const [infraStatus, setInfraStatus] = useState(null);
  const [probingInfra, setProbingInfra] = useState(false);
  const [rotatingSigning, setRotatingSigning] = useState(false);
  const [signingScope, setSigningScope] = useState("all");
  const anomalyIndexSet = useMemo(() => new Set(anomalyIndices), [anomalyIndices]);

  const timeSeriesData = useMemo(() => {
    return normalizedPoints.map((point, index) => ({
      index,
      time: point.time,
      flux: point.flux,
      isAnomaly: anomalyIndexSet.has(index),
      score: scores[index] ?? null,
    }));
  }, [normalizedPoints, anomalyIndexSet, scores]);

  const anomalyPoints = useMemo(() => timeSeriesData.filter((item) => item.isAnomaly), [timeSeriesData]);
  const scoreSeries = useMemo(() => scores.map((value, index) => ({ index, score: value })), [scores]);
  const heatmapSeries = useMemo(() => xaiHeatmap.map((value, index) => ({ index, heat: value })), [xaiHeatmap]);
  const ensembleSeries = useMemo(() => ensembleConfidence.map((value, index) => ({ index, confidence: value })), [ensembleConfidence]);
  const displayTimeSeriesData = useMemo(() => downsampleByIndex(timeSeriesData, MAX_CHART_POINTS), [timeSeriesData]);
  const displayScoreSeries = useMemo(() => downsampleByIndex(scoreSeries, MAX_CHART_POINTS), [scoreSeries]);
  const displayHeatmapSeries = useMemo(() => downsampleByIndex(heatmapSeries, MAX_CHART_POINTS), [heatmapSeries]);
  const displayEnsembleSeries = useMemo(() => downsampleByIndex(ensembleSeries, MAX_CHART_POINTS), [ensembleSeries]);
  const latentScatterData = useMemo(
    () => latentPoints.map((point) => ({ x: point.x, y: point.y, cluster: point.cluster })),
    [latentPoints]
  );

  const updateModelSelection = (modelName) => {
    setSelectedModels((prev) => (prev.includes(modelName) ? prev.filter((item) => item !== modelName) : [...prev, modelName]));
  };

  const fetchSummary = async (targetDatasetId) => {
    const response = await api.get(`/datasets/${targetDatasetId}/summary`, {
      params: { recent_only: recentOnly, recent_years: recentYears },
    });
    setDatasetMeta(response.data.meta);
    setDatasetStats(response.data.stats);
  };

  const fetchHistory = async (targetDatasetId, nextOffset = 0, nextModel = historyModelFilter) => {
    const modelNameParam = nextModel === "all" ? undefined : nextModel;
    const response = await api.get("/results", {
      params: {
        dataset_id: targetDatasetId,
        model_name: modelNameParam,
        limit: historyLimit,
        offset: nextOffset,
      },
    });
    setHistoryItems(response.data.results);
    setHistoryTotal(response.data.total);
    setHistoryOffset(nextOffset);
  };

  const applyDatasetResponse = async (data) => {
    setDatasetId(data.dataset_id);
    setFilename(data.filename);
    setPointsCount(data.points.length);
    setNormalizedPoints(data.normalized_points);
    setDatasetMeta(data.meta);
    setTrainingSummary([]);
    setComparisonSummary([]);
    setBestModel("");
    setScores([]);
    setXaiHeatmap([]);
    setEnsembleConfidence([]);
    setAnomalyIndices([]);
    setInsightSummary("");
    setInsightBackend("");
    setLatentPoints([]);
    setResultId("");
    setJobInfo(null);
    setActiveTab("pipeline");
    await fetchSummary(data.dataset_id);
    await fetchHistory(data.dataset_id, 0, "all");
  };

  const onUpload = async () => {
    if (!selectedFile) {
      setError("Select a CSV file first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const response = await api.post(`/upload?recent_only=${recentOnly}&recent_years=${recentYears}`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      await applyDatasetResponse(response.data);
    } catch (uploadError) {
      setError(uploadError?.response?.data?.detail || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  const onFetchNasa = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await api.get(`/fetch/nasa-fireball?recent_years=${recentYears}`);
      await applyDatasetResponse(response.data);
    } catch (fetchError) {
      setError(fetchError?.response?.data?.detail || "NASA fetch failed");
    } finally {
      setLoading(false);
    }
  };

  const onTrain = async () => {
    if (!datasetId) {
      setError("Upload a dataset first.");
      return;
    }
    if (selectedModels.length === 0) {
      setError("Select at least one model.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await api.post("/train", {
        dataset_id: datasetId,
        models: selectedModels,
        epochs: 20,
        recent_only: recentOnly,
        recent_years: recentYears,
      });
      setTrainingSummary(response.data.training);
      if (!selectedModels.includes(detectModel)) {
        setDetectModel(selectedModels[0]);
      }
    } catch (trainError) {
      setError(trainError?.response?.data?.detail || "Training failed");
    } finally {
      setLoading(false);
    }
  };

  const onTrainAsync = async () => {
    if (!datasetId) {
      setError("Upload a dataset first.");
      return;
    }
    if (selectedModels.length === 0) {
      setError("Select at least one model.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await api.post("/train/async", {
        dataset_id: datasetId,
        models: selectedModels,
        epochs: 20,
        recent_only: recentOnly,
        recent_years: recentYears,
      });
      setJobInfo({ id: response.data.job_id, type: "train", status: response.data.status });
    } catch (trainError) {
      setError(trainError?.response?.data?.detail || "Async training failed to start");
    } finally {
      setLoading(false);
    }
  };

  const onDetect = async () => {
    if (!datasetId) {
      setError("Upload a dataset first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await api.post("/detect", {
        dataset_id: datasetId,
        model_name: detectModel,
        epochs: 20,
        threshold_percentile: thresholdPercentile,
        recent_only: recentOnly,
        recent_years: recentYears,
      });
      setScores(response.data.scores);
      setXaiHeatmap(response.data.xai_heatmap || []);
      setAnomalyIndices(response.data.anomaly_indices);
      setResultId(response.data.result_id);
      setInsightSummary(response.data.insight_summary || "");
      setInsightBackend(response.data.insight_backend || "");
      await fetchHistory(datasetId, 0);
    } catch (detectError) {
      setError(detectError?.response?.data?.detail || "Detection failed");
    } finally {
      setLoading(false);
    }
  };

  const onEnsembleDiscover = async () => {
    if (!datasetId) {
      setError("Upload a dataset first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await api.post("/ensemble/discover", {
        dataset_id: datasetId,
        models: selectedModels.length > 0 ? selectedModels : modelOptions,
        threshold_percentile: thresholdPercentile,
        recent_only: recentOnly,
        recent_years: recentYears,
        batch_size: 512,
      });
      setEnsembleConfidence(response.data.confidence_index || []);
      setAnomalyIndices(response.data.anomaly_indices || []);
      setActiveTab("charts");
    } catch (ensembleError) {
      setError(ensembleError?.response?.data?.detail || "Ensemble discovery failed");
    } finally {
      setLoading(false);
    }
  };

  const onCompare = async () => {
    if (!datasetId) {
      setError("Upload a dataset first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await api.post("/compare", {
        dataset_id: datasetId,
        models: selectedModels.length > 0 ? selectedModels : modelOptions,
        epochs: 20,
        threshold_percentile: thresholdPercentile,
        recent_only: recentOnly,
        recent_years: recentYears,
      });
      setComparisonSummary(response.data.comparisons);
      setBestModel(response.data.best_model);
      await fetchHistory(datasetId, 0);
    } catch (compareError) {
      setError(compareError?.response?.data?.detail || "Comparison failed");
    } finally {
      setLoading(false);
    }
  };

  const onFetchLatentProjection = async () => {
    if (!datasetId) {
      setError("Upload a dataset first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await api.post("/latent/projection", {
        dataset_id: datasetId,
        recent_only: recentOnly,
        recent_years: recentYears,
        sample_limit: 1200,
      });
      setLatentPoints(response.data.points || []);
      setLatentMethod(response.data.method || "umap-style");
      setActiveTab("charts");
    } catch (latentError) {
      setError(latentError?.response?.data?.detail || "Latent projection failed");
    } finally {
      setLoading(false);
    }
  };

  const fetchAgentPanels = useCallback(async (targetDatasetId = datasetId) => {
    const [activityResponse, discoveryResponse, summaryResponse, rlResponse] = await Promise.all([
      api.get("/agent/activity-feed", { params: { limit: 60 } }),
      api.get("/discoveries", { params: { dataset_id: targetDatasetId || undefined, limit: 100 } }),
      api.get("/discoveries/summary", { params: { dataset_id: targetDatasetId || undefined } }),
      api.get("/rl/trainer/stats"),
    ]);
    setAgentActivities(activityResponse.data.activities || []);
    setDiscoveries(discoveryResponse.data.discoveries || []);
    setDiscoverySummary(summaryResponse.data || null);
    setRlStats(rlResponse.data || null);
  }, [datasetId]);

  const fetchInfraStatus = useCallback(async (liveProbe = false) => {
    const response = await api.get("/infra/status", { params: { live_probe: liveProbe } });
    setInfraStatus(response.data || null);
  }, []);

  const onRunAgentCycle = async () => {
    if (!datasetId) {
      setError("Upload a dataset first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await api.post("/agent/run-cycle", {
        dataset_id: datasetId,
        models: selectedModels.length > 0 ? selectedModels : modelOptions,
        epochs: 2,
        recent_only: recentOnly,
        recent_years: recentYears,
        use_gpu: true,
        batch_size: 512,
      });
      setAgentOutput(response.data);
      setReasoningText(response.data.reasoning || "");
      setReasoningCursor(0);
      setAnomalyIndices(response.data.anomaly_indices || []);
      setEnsembleConfidence(response.data.confidence_index || []);
      await fetchAgentPanels(datasetId);
      setActiveTab("agent");
    } catch (agentError) {
      setError(agentError?.response?.data?.detail || "Agent cycle failed");
    } finally {
      setLoading(false);
    }
  };

  const onFeedbackDiscovery = async (discoveryId, thumbsUp) => {
    setLoading(true);
    setError("");
    try {
      await api.post("/discoveries/feedback", {
        discovery_id: discoveryId,
        thumbs_up: thumbsUp,
        notes: thumbsUp ? "confirmed by expert" : "likely false positive",
      });
      await fetchAgentPanels(datasetId);
    } catch (feedbackError) {
      setError(feedbackError?.response?.data?.detail || "Feedback submit failed");
    } finally {
      setLoading(false);
    }
  };

  const onCompareAsync = async () => {
    if (!datasetId) {
      setError("Upload a dataset first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await api.post("/compare/async", {
        dataset_id: datasetId,
        models: selectedModels.length > 0 ? selectedModels : modelOptions,
        epochs: 20,
        threshold_percentile: thresholdPercentile,
        recent_only: recentOnly,
        recent_years: recentYears,
      });
      setJobInfo({ id: response.data.job_id, type: "compare", status: response.data.status });
    } catch (compareError) {
      setError(compareError?.response?.data?.detail || "Async comparison failed to start");
    } finally {
      setLoading(false);
    }
  };

  const onDownload = async () => {
    if (!resultId) {
      return;
    }
    const response = await api.get(`/results/${resultId}/download`, {
      responseType: "blob",
    });
    const url = URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `${resultId}_scores.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const onLogin = async () => {
    if (!loginEmail.trim() || !loginPassword.trim()) {
      setError("Enter email and password.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await api.post("/auth/login", {
        email: loginEmail.trim(),
        password: loginPassword,
      });
      localStorage.setItem("astro_token", response.data.access_token);
      setUserProfile(response.data.user);
      setIsAuthenticated(true);
      setScreen("dashboard");
    } catch (loginError) {
      setError(loginError?.response?.data?.detail || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  const onLogout = () => {
    localStorage.removeItem("astro_token");
    setIsAuthenticated(false);
    setUserProfile(null);
    setScreen("home");
    setLoginEmail("");
    setLoginPassword("");
  };

  useEffect(() => {
    if (!isAuthenticated && screen === "dashboard") {
      setScreen("login");
    }
  }, [isAuthenticated, screen]);

  useEffect(() => {
    const token = localStorage.getItem("astro_token");
    if (!token) {
      return;
    }
    api
      .get("/auth/me")
      .then((response) => {
        setUserProfile(response.data);
        setIsAuthenticated(true);
        setScreen("dashboard");
      })
      .catch(() => {
        localStorage.removeItem("astro_token");
        setIsAuthenticated(false);
        setUserProfile(null);
      });
  }, []);

  useEffect(() => {
    if (!jobInfo?.id || (jobInfo.status !== "queued" && jobInfo.status !== "running")) {
      return undefined;
    }
    const intervalId = setInterval(async () => {
      try {
        const response = await api.get(`/jobs/${jobInfo.id}`);
        const job = response.data;
        setJobInfo({ id: job.id, type: job.type, status: job.status });
        if (job.status === "completed") {
          if (job.type === "train") {
            setTrainingSummary(job.result.training || []);
            if (!selectedModels.includes(detectModel) && selectedModels.length > 0) {
              setDetectModel(selectedModels[0]);
            }
          }
          if (job.type === "compare") {
            setComparisonSummary(job.result.comparisons || []);
            setBestModel(job.result.best_model || "");
            if (datasetId) {
              const historyResponse = await api.get("/results", {
                params: {
                  dataset_id: datasetId,
                  model_name: historyModelFilter === "all" ? undefined : historyModelFilter,
                  limit: historyLimit,
                  offset: 0,
                },
              });
              setHistoryItems(historyResponse.data.results);
              setHistoryTotal(historyResponse.data.total);
              setHistoryOffset(0);
            }
          }
        }
        if (job.status === "failed") {
          setError(job.error || "Background job failed");
        }
      } catch (pollError) {
        setError(pollError?.response?.data?.detail || "Job polling failed");
      }
    }, 2000);
    return () => clearInterval(intervalId);
  }, [jobInfo, datasetId, selectedModels, detectModel, historyModelFilter, historyLimit]);

  useEffect(() => {
    if (!isAuthenticated) {
      return;
    }
    api
      .get("/platform/capabilities")
      .then((response) => setCapabilities(response.data))
      .catch(() => {});
    api
      .get("/streams/live?limit=8")
      .then((response) => setLiveEvents(response.data.events || []))
      .catch(() => {});
  }, [isAuthenticated]);

  useEffect(() => {
    if (!isAuthenticated) {
      return;
    }
    fetchAgentPanels().catch(() => {});
    fetchInfraStatus(false).catch(() => {});
  }, [isAuthenticated, fetchAgentPanels, fetchInfraStatus]);

  useEffect(() => {
    if (!reasoningText) {
      return undefined;
    }
    const timer = setInterval(() => {
      setReasoningCursor((prev) => {
        if (prev >= reasoningText.length) {
          clearInterval(timer);
          return prev;
        }
        return prev + 3;
      });
    }, 24);
    return () => clearInterval(timer);
  }, [reasoningText]);

  const renderHome = () => (
    <div className="relative min-h-screen overflow-hidden bg-slate-950 text-white">
      <div className="absolute -left-20 top-16 h-72 w-72 rounded-full bg-cyan-500/30 blur-3xl animate-float-slow" />
      <div className="absolute right-4 top-20 h-80 w-80 rounded-full bg-fuchsia-500/30 blur-3xl animate-float-medium" />
      <div className="absolute bottom-0 left-1/3 h-96 w-96 rounded-full bg-indigo-500/30 blur-3xl animate-float-fast" />
      <div className="relative mx-auto flex min-h-screen max-w-6xl flex-col items-center justify-center px-6 text-center">
        <span className="rounded-full border border-cyan-300/40 bg-cyan-500/10 px-4 py-1 text-xs tracking-wider text-cyan-200">
          UNSUPERVISED DEEP LEARNING PLATFORM
        </span>
        <h1 className="mt-6 text-4xl font-bold leading-tight md:text-6xl">
          Astronomical <span className="text-cyan-300">Anomaly Discovery</span>
        </h1>
        <p className="mt-4 max-w-3xl text-sm text-slate-200 md:text-lg">
          Upload light curves, run Autoencoder/VAE/Transformer models, detect anomalies, compare performance, and export research-ready outputs.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
          <button
            className="rounded-lg bg-cyan-400 px-6 py-3 font-semibold text-slate-900 transition hover:scale-105"
            onClick={() => setScreen(isAuthenticated ? "dashboard" : "login")}
          >
            Start Platform
          </button>
          <button
            className="rounded-lg border border-slate-500 px-6 py-3 font-semibold transition hover:bg-slate-800"
            onClick={() => setScreen("login")}
          >
            Login
          </button>
        </div>
      </div>
    </div>
  );

  const renderLogin = () => (
    <div className="min-h-screen bg-slate-950 px-6 py-10 text-white">
      <div className="mx-auto max-w-4xl">
        <button className="mb-6 rounded border border-slate-600 px-3 py-1 text-sm" onClick={() => setScreen("home")}>
          Back
        </button>
        <div className="grid gap-8 md:grid-cols-2">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-8">
            <h2 className="text-2xl font-semibold">Welcome Back</h2>
            <p className="mt-2 text-sm text-slate-300">Login to access your main dashboard and anomaly research tools.</p>
            <div className="mt-6 space-y-3">
              <input
                className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2"
                placeholder="Email"
                value={loginEmail}
                onChange={(event) => setLoginEmail(event.target.value)}
              />
              <input
                className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2"
                placeholder="Password"
                type="password"
                value={loginPassword}
                onChange={(event) => setLoginPassword(event.target.value)}
              />
              <button className="w-full rounded bg-cyan-400 px-4 py-2 font-semibold text-slate-900" onClick={onLogin}>
                {loading ? "Signing in..." : "Login to Dashboard"}
              </button>
              <div className="rounded border border-slate-700 bg-slate-950 p-2 text-xs text-slate-300">
                Demo users: admin@astro.local / admin123, researcher@astro.local / research123
              </div>
            </div>
          </div>
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-8">
            <h3 className="text-xl font-semibold">Project Highlights</h3>
            <ul className="mt-4 space-y-3 text-sm text-slate-300">
              <li>End-to-end light-curve pipeline</li>
              <li>Real-time anomaly visualization</li>
              <li>Model comparison and history tracking</li>
              <li>NASA fireball ingestion support</li>
              <li>Async jobs for heavy workloads</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );

  const renderDashboard = () => (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="border-b border-slate-800 bg-slate-900/80">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-xl font-semibold md:text-2xl">Main Dashboard</h1>
            <p className="text-xs text-slate-300 md:text-sm">Anomaly analytics workspace for your astronomical research project</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden rounded border border-slate-700 bg-slate-950 px-3 py-1 text-xs text-slate-300 md:block">
              {userProfile?.full_name || "User"} · {userProfile?.role || "guest"}
            </div>
            <button className="rounded border border-slate-600 px-3 py-1 text-sm" onClick={() => setScreen("home")}>
              Home
            </button>
            <button className="rounded bg-rose-500 px-3 py-1 text-sm font-semibold text-white" onClick={onLogout}>
              Logout
            </button>
          </div>
        </div>
      </div>
      <div className="mx-auto grid max-w-7xl gap-6 px-6 py-8 lg:grid-cols-[220px_1fr]">
        <aside className="rounded-xl border border-slate-800 bg-slate-900 p-3">
          <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">Navigation</div>
          {dashboardTabs.map((tab) => (
            <button
              key={tab.id}
              className={`mb-2 w-full rounded px-3 py-2 text-left text-sm transition ${
                activeTab === tab.id ? "bg-cyan-500 text-slate-900 font-semibold" : "bg-slate-950 text-slate-300 hover:bg-slate-800"
              }`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </aside>
        <div>
        {activeTab === "pipeline" && (
        <>
        <div className="mt-2 grid gap-6 lg:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="text-lg font-medium">1. Upload Data</h2>
            <label className="mt-3 flex items-center gap-2 text-sm">
              <input type="checkbox" checked={recentOnly} onChange={(event) => setRecentOnly(event.target.checked)} />
              <span>Use recent {recentYears} years only</span>
            </label>
            <div className="mt-2 flex items-center gap-2">
              <span className="text-xs text-slate-400">Years</span>
              <input
                className="w-16 rounded border border-slate-700 bg-slate-950 p-1 text-sm"
                type="number"
                min={1}
                max={10}
                value={recentYears}
                onChange={(event) => setRecentYears(Number(event.target.value) || 2)}
              />
            </div>
            <input
              className="mt-3 block w-full rounded border border-slate-700 bg-slate-950 p-2 text-sm"
              type="file"
              accept=".csv"
              onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
            />
            <button className="mt-3 w-full rounded bg-cyan-500 px-4 py-2 font-medium text-slate-900 disabled:opacity-50" onClick={onUpload} disabled={loading}>
              Upload CSV
            </button>
            <button className="mt-3 w-full rounded bg-indigo-500 px-4 py-2 font-medium text-white disabled:opacity-50" onClick={onFetchNasa} disabled={loading}>
              Fetch NASA Fireball Data (Recent {recentYears} Years)
            </button>
            <div className="mt-2 text-xs text-slate-400">{filename ? `Dataset: ${filename}` : "No dataset uploaded"}</div>
            <div className="text-xs text-slate-400">{datasetId ? `Dataset ID: ${datasetId}` : ""}</div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="text-lg font-medium">2. Train Models</h2>
            <div className="mt-3 space-y-2">
              {modelOptions.map((model) => (
                <label key={model} className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={selectedModels.includes(model)} onChange={() => updateModelSelection(model)} />
                  <span>{model}</span>
                </label>
              ))}
            </div>
            <button className="mt-3 w-full rounded bg-emerald-500 px-4 py-2 font-medium text-slate-900 disabled:opacity-50" onClick={onTrain} disabled={loading}>
              Run Training
            </button>
            <button className="mt-3 w-full rounded border border-emerald-500 px-4 py-2 font-medium text-emerald-300 disabled:opacity-50" onClick={onTrainAsync} disabled={loading}>
              Run Training Async
            </button>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="text-lg font-medium">3. Detect Anomalies</h2>
            <div className="mt-3 text-xs text-slate-400">Threshold percentile: {thresholdPercentile}%</div>
            <input className="mt-1 w-full" type="range" min={80} max={99} value={thresholdPercentile} onChange={(event) => setThresholdPercentile(Number(event.target.value))} />
            <select className="mt-3 w-full rounded border border-slate-700 bg-slate-950 p-2 text-sm" value={detectModel} onChange={(event) => setDetectModel(event.target.value)}>
              {modelOptions.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
            <button className="mt-3 w-full rounded bg-fuchsia-500 px-4 py-2 font-medium text-slate-900 disabled:opacity-50" onClick={onDetect} disabled={loading}>
              Run Detection
            </button>
            <button className="mt-3 w-full rounded bg-amber-500 px-4 py-2 font-medium text-slate-900 disabled:opacity-50" onClick={onCompare} disabled={loading}>
              Compare All Selected Models
            </button>
            <button className="mt-3 w-full rounded bg-cyan-500 px-4 py-2 font-medium text-slate-900 disabled:opacity-50" onClick={onEnsembleDiscover} disabled={loading}>
              Ensemble Discovery Confidence
            </button>
            <button className="mt-3 w-full rounded bg-violet-500 px-4 py-2 font-medium text-white disabled:opacity-50" onClick={onRunAgentCycle} disabled={loading}>
              Run Autonomous Agent
            </button>
            <button className="mt-3 w-full rounded bg-sky-500 px-4 py-2 font-medium text-slate-900 disabled:opacity-50" onClick={onFetchLatentProjection} disabled={loading}>
              Build Latent Space Map
            </button>
            <button className="mt-3 w-full rounded border border-amber-500 px-4 py-2 font-medium text-amber-300 disabled:opacity-50" onClick={onCompareAsync} disabled={loading}>
              Compare Async
            </button>
            <button className="mt-3 w-full rounded border border-slate-600 px-4 py-2 font-medium disabled:opacity-50" onClick={onDownload} disabled={!resultId || loading}>
              Download Scores CSV
            </button>
          </div>
        </div>
        {loading && <div className="mt-4 rounded-md border border-cyan-600 bg-cyan-950 px-4 py-2 text-sm">Processing request. Training and detection can take a few seconds.</div>}
        {jobInfo && <div className="mt-4 rounded-md border border-violet-600 bg-violet-950 px-4 py-2 text-sm">Background job {jobInfo.id} ({jobInfo.type}) status: {jobInfo.status}</div>}
        {error && <div className="mt-4 rounded-md border border-red-600 bg-red-950 px-4 py-2 text-sm text-red-200">{error}</div>}
        {insightSummary && (
          <div className="mt-4 rounded-md border border-violet-600 bg-violet-950 px-4 py-2 text-sm">
            <div className="text-xs text-violet-300">SLM Insight · {insightBackend || "template"}</div>
            <div className="mt-1 text-violet-100">{insightSummary}</div>
          </div>
        )}
        <div className="mt-6 grid gap-4 md:grid-cols-4">
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-3">
            <div className="text-xs text-slate-400">Total Points</div>
            <div className="text-xl font-semibold">{datasetStats?.points ?? 0}</div>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-3">
            <div className="text-xs text-slate-400">Mean Flux</div>
            <div className="text-xl font-semibold">{datasetStats ? datasetStats.mean_flux.toFixed(4) : "0.0000"}</div>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-3">
            <div className="text-xs text-slate-400">Std Flux</div>
            <div className="text-xl font-semibold">{datasetStats ? datasetStats.std_flux.toFixed(4) : "0.0000"}</div>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-3">
            <div className="text-xs text-slate-400">Detected Anomalies</div>
            <div className="text-xl font-semibold">{anomalyIndices.length}</div>
          </div>
        </div>
        </>
        )}
        {activeTab === "charts" && (
        <div className="mt-2 grid gap-6 lg:grid-cols-2">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="text-lg font-medium">Normalized Time-Series</h2>
            <div className="mt-4 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={displayTimeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#cbd5e1" />
                  <YAxis stroke="#cbd5e1" />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="flux" stroke="#22d3ee" dot={false} name="Normalized Flux" />
                  <Scatter data={anomalyPoints} fill="#ef4444" name="Anomalies" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="text-lg font-medium">Anomaly Score</h2>
            <div className="mt-4 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={displayScoreSeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="index" stroke="#cbd5e1" />
                  <YAxis stroke="#cbd5e1" />
                  <Tooltip />
                  <Line type="monotone" dataKey="score" stroke="#a78bfa" dot={false} name="Anomaly Score" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="text-lg font-medium">XAI Heatmap Trigger</h2>
            <div className="mt-4 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={displayHeatmapSeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="index" stroke="#cbd5e1" />
                  <YAxis stroke="#cbd5e1" />
                  <Tooltip />
                  <Line type="monotone" dataKey="heat" stroke="#f43f5e" dot={false} name="XAI Heat" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="text-lg font-medium">Unified Confidence Index</h2>
            <div className="mt-4 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={displayEnsembleSeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="index" stroke="#cbd5e1" />
                  <YAxis stroke="#cbd5e1" />
                  <Tooltip />
                  <Line type="monotone" dataKey="confidence" stroke="#22c55e" dot={false} name="Confidence" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 lg:col-span-2">
            <h2 className="text-lg font-medium">Latent Space Explorer (2D)</h2>
            <div className="mt-1 text-xs text-slate-400">{latentPoints.length > 0 ? `${latentPoints.length} embedded windows · ${latentMethod}` : "Run Build Latent Space Map from Pipeline tab"}</div>
            <div className="mt-4 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis type="number" dataKey="x" stroke="#cbd5e1" name="x" />
                  <YAxis type="number" dataKey="y" stroke="#cbd5e1" name="y" />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                  <Scatter data={latentScatterData} fill="#38bdf8" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
        )}
        {activeTab === "comparison" && (
        <div className="mt-2 rounded-xl border border-slate-800 bg-slate-900 p-4">
          <h2 className="text-lg font-medium">Model Comparison</h2>
          <div className="mt-1 text-xs text-slate-400">{bestModel ? `Best model: ${bestModel}` : "Run compare to identify best model"}</div>
          <div className="mt-3 overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="border-b border-slate-700 text-slate-300">
                <tr>
                  <th className="px-3 py-2">Model</th>
                  <th className="px-3 py-2">Final Training Loss</th>
                  <th className="px-3 py-2">Anomaly Count</th>
                  <th className="px-3 py-2">Threshold</th>
                </tr>
              </thead>
              <tbody>
                {comparisonSummary.length === 0 && trainingSummary.length === 0 && (
                  <tr>
                    <td className="px-3 py-3 text-slate-400" colSpan={4}>
                      No training results yet
                    </td>
                  </tr>
                )}
                {comparisonSummary.length > 0 &&
                  comparisonSummary.map((row) => (
                    <tr key={row.model_name} className="border-b border-slate-800">
                      <td className="px-3 py-2">{row.model_name}</td>
                      <td className="px-3 py-2">{row.final_loss}</td>
                      <td className="px-3 py-2">{row.anomaly_count}</td>
                      <td className="px-3 py-2">{row.threshold.toFixed(6)}</td>
                    </tr>
                  ))}
                {comparisonSummary.length === 0 &&
                  trainingSummary.map((row) => (
                    <tr key={row.model_name} className="border-b border-slate-800">
                      <td className="px-3 py-2">{row.model_name}</td>
                      <td className="px-3 py-2">{row.final_loss}</td>
                      <td className="px-3 py-2">-</td>
                      <td className="px-3 py-2">-</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
        )}
        {activeTab === "innovation" && (
        <div className="mt-2 grid gap-6 lg:grid-cols-2">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="text-lg font-medium">Integrated Vertical Features</h2>
            <ul className="mt-3 space-y-2 text-sm text-slate-300">
              {capabilities.vertical?.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="text-lg font-medium">Integrated Horizontal Features</h2>
            <ul className="mt-3 space-y-2 text-sm text-slate-300">
              {capabilities.horizontal?.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 lg:col-span-2">
            <h2 className="text-lg font-medium">Live ZTF/LSST Stream Preview</h2>
            <div className="mt-3 overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="border-b border-slate-700 text-slate-300">
                  <tr>
                    <th className="px-3 py-2">Event</th>
                    <th className="px-3 py-2">Survey</th>
                    <th className="px-3 py-2">RA</th>
                    <th className="px-3 py-2">DEC</th>
                    <th className="px-3 py-2">Mag</th>
                  </tr>
                </thead>
                <tbody>
                  {liveEvents.map((row) => (
                    <tr key={row.event_id} className="border-b border-slate-800">
                      <td className="px-3 py-2">{row.event_id}</td>
                      <td className="px-3 py-2">{row.survey}</td>
                      <td className="px-3 py-2">{row.ra}</td>
                      <td className="px-3 py-2">{row.dec}</td>
                      <td className="px-3 py-2">{row.magnitude}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        )}
        {activeTab === "history" && (
        <div className="mt-2 rounded-xl border border-slate-800 bg-slate-900 p-4">
          <h2 className="text-lg font-medium">Results History</h2>
          <div className="mt-3 flex items-center gap-3">
            <select
              className="rounded border border-slate-700 bg-slate-950 p-2 text-sm"
              value={historyModelFilter}
              onChange={async (event) => {
                const nextModel = event.target.value;
                setHistoryModelFilter(nextModel);
                if (datasetId) {
                  await fetchHistory(datasetId, 0, nextModel);
                }
              }}
            >
              <option value="all">all models</option>
              {modelOptions.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
            <button
              className="rounded border border-slate-700 px-3 py-2 text-sm"
              onClick={() => datasetId && fetchHistory(datasetId, Math.max(0, historyOffset - historyLimit))}
              disabled={historyOffset === 0 || !datasetId}
            >
              Previous
            </button>
            <button
              className="rounded border border-slate-700 px-3 py-2 text-sm"
              onClick={() => datasetId && fetchHistory(datasetId, historyOffset + historyLimit)}
              disabled={historyOffset + historyLimit >= historyTotal || !datasetId}
            >
              Next
            </button>
            <div className="text-xs text-slate-400">
              {historyTotal === 0 ? "0 results" : `${historyOffset + 1}-${Math.min(historyOffset + historyLimit, historyTotal)} of ${historyTotal}`}
            </div>
          </div>
          <div className="mt-3 overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="border-b border-slate-700 text-slate-300">
                <tr>
                  <th className="px-3 py-2">Model</th>
                  <th className="px-3 py-2">Anomalies</th>
                  <th className="px-3 py-2">Threshold</th>
                  <th className="px-3 py-2">Feedback</th>
                  <th className="px-3 py-2">Created</th>
                </tr>
              </thead>
              <tbody>
                {historyItems.length === 0 && (
                  <tr className="border-b border-slate-800">
                    <td className="px-3 py-3 text-slate-400" colSpan={5}>
                      No result history yet
                    </td>
                  </tr>
                )}
                {historyItems.map((row) => (
                  <tr key={row.id} className="border-b border-slate-800">
                    <td className="px-3 py-2">{row.model_name}</td>
                    <td className="px-3 py-2">{row.anomaly_indices.length}</td>
                    <td className="px-3 py-2">{row.threshold.toFixed(6)}</td>
                    <td className="px-3 py-2">
                      <select
                        className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs"
                        value={row.feedback || "unreviewed"}
                        onChange={async (event) => {
                          const nextFeedback = event.target.value;
                          try {
                            await api.post(`/results/${row.id}/feedback`, { feedback: nextFeedback });
                            await fetchHistory(datasetId, historyOffset, historyModelFilter);
                          } catch (feedbackError) {
                            setError(feedbackError?.response?.data?.detail || "Feedback update failed");
                          }
                        }}
                      >
                        <option value="unreviewed">unreviewed</option>
                        <option value="real_discovery">real_discovery</option>
                        <option value="instrumental_noise">instrumental_noise</option>
                      </select>
                    </td>
                    <td className="px-3 py-2">{new Date(row.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        )}
        {activeTab === "agent" && (
        <div className="mt-2 grid gap-6 lg:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 lg:col-span-2">
            <h2 className="text-lg font-medium">Reasoning Terminal</h2>
            <div className="mt-2 text-xs text-slate-400">SLM chain-of-thought stream</div>
            <div className="mt-3 h-64 overflow-auto rounded border border-slate-700 bg-black p-3 font-mono text-xs leading-6 text-emerald-300">
              {(reasoningText || "No reasoning yet. Run Autonomous Agent.").slice(0, reasoningCursor)}
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-4">
              <div className="rounded border border-slate-700 bg-slate-950 p-2 text-xs">
                <div className="text-slate-400">Confidence</div>
                <div className="text-base font-semibold">{agentOutput?.confidence ? agentOutput.confidence.toFixed(4) : "-"}</div>
              </div>
              <div className="rounded border border-slate-700 bg-slate-950 p-2 text-xs">
                <div className="text-slate-400">Policy Threshold</div>
                <div className="text-base font-semibold">{rlStats?.policy?.threshold_percentile?.toFixed?.(3) || "-"}</div>
              </div>
              <div className="rounded border border-slate-700 bg-slate-950 p-2 text-xs">
                <div className="text-slate-400">Precision Proxy</div>
                <div className="text-base font-semibold">{rlStats?.feedback?.precision_proxy?.toFixed?.(3) || "-"}</div>
              </div>
              <div className="rounded border border-slate-700 bg-slate-950 p-2 text-xs">
                <div className="text-slate-400">Triggered Channels</div>
                <div className="text-base font-semibold">{agentOutput?.triggered_channels?.length ?? 0}</div>
              </div>
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="text-lg font-medium">Discoveries Summary</h2>
            <div className="mt-3 space-y-2 text-sm">
              <div className="flex items-center justify-between rounded border border-slate-700 bg-slate-950 px-3 py-2"><span>Total</span><span>{discoverySummary?.total ?? 0}</span></div>
              <div className="flex items-center justify-between rounded border border-slate-700 bg-slate-950 px-3 py-2"><span>Confirmed</span><span>{discoverySummary?.confirmed ?? 0}</span></div>
              <div className="flex items-center justify-between rounded border border-slate-700 bg-slate-950 px-3 py-2"><span>Candidate</span><span>{discoverySummary?.candidate ?? 0}</span></div>
              <div className="flex items-center justify-between rounded border border-slate-700 bg-slate-950 px-3 py-2"><span>Rejected</span><span>{discoverySummary?.rejected ?? 0}</span></div>
              <div className="flex items-center justify-between rounded border border-slate-700 bg-slate-950 px-3 py-2"><span>Discovery Yield</span><span>{discoverySummary?.discovery_yield?.toFixed?.(3) || "0.000"}</span></div>
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 lg:col-span-3">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-medium">Infrastructure Backends</h2>
              <div className="flex items-center gap-2">
                <button
                  className="rounded border border-slate-600 px-3 py-1 text-xs disabled:opacity-50"
                  disabled={probingInfra}
                  onClick={async () => {
                    setProbingInfra(true);
                    try {
                      await fetchInfraStatus(true);
                    } finally {
                      setProbingInfra(false);
                    }
                  }}
                >
                  {probingInfra ? "Probing..." : "Run Live Probe"}
                </button>
                <select
                  className="rounded border border-slate-600 bg-slate-950 px-2 py-1 text-xs"
                  value={signingScope}
                  onChange={(event) => setSigningScope(event.target.value)}
                >
                  <option value="all">all</option>
                  <option value="slm">slm</option>
                  <option value="vector">vector</option>
                </select>
                <button
                  className="rounded border border-violet-500 px-3 py-1 text-xs disabled:opacity-50"
                  disabled={rotatingSigning}
                  onClick={async () => {
                    setRotatingSigning(true);
                    try {
                      await api.post("/infra/signing/rotate", { scope: signingScope });
                      await fetchInfraStatus(false);
                    } finally {
                      setRotatingSigning(false);
                    }
                  }}
                >
                  {rotatingSigning ? "Rotating..." : "Rotate Key"}
                </button>
              </div>
            </div>
            <div className="mt-3 grid gap-3 md:grid-cols-3 text-xs">
              <div className="rounded border border-slate-700 bg-slate-950 p-3">
                <div className="text-slate-400">SLM Backend</div>
                <div className="font-semibold">{infraStatus?.agent_backends?.slm?.backend || "-"}</div>
                <div className="text-slate-500">{infraStatus?.triton_http_url || "no triton url"}</div>
                <div className="text-slate-500">signing keys: {infraStatus?.signing?.slm?.available_keys ?? 0}</div>
              </div>
              <div className="rounded border border-slate-700 bg-slate-950 p-3">
                <div className="text-slate-400">Vector Provider</div>
                <div className="font-semibold">{infraStatus?.agent_backends?.vector_store?.provider || "-"}</div>
                <div className="text-slate-500">{infraStatus?.pinecone_index_host || "no pinecone host"}</div>
                <div className="text-slate-500">signing keys: {infraStatus?.signing?.vector_store?.available_keys ?? 0}</div>
              </div>
              <div className="rounded border border-slate-700 bg-slate-950 p-3">
                <div className="text-slate-400">Live Probe</div>
                <div className="font-semibold">triton: {infraStatus?.live_probe?.triton || "skipped"}</div>
                <div className="font-semibold">pinecone: {infraStatus?.live_probe?.pinecone || "skipped"}</div>
                <div className="text-slate-500">monitor interval: {infraStatus?.health_monitor_interval || "-"}</div>
              </div>
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 lg:col-span-2">
            <h2 className="text-lg font-medium">Agent Activity Feed</h2>
            <div className="mt-3 h-72 overflow-auto rounded border border-slate-700 bg-slate-950">
              {agentActivities.map((item) => (
                <div key={item.id} className="border-b border-slate-800 px-3 py-2 text-xs">
                  <div className="font-semibold text-cyan-300">{item.stage}</div>
                  <div className="text-slate-200">{item.message}</div>
                  <div className="text-slate-500">{new Date(item.created_at).toLocaleString()}</div>
                </div>
              ))}
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="text-lg font-medium">RL Trainer</h2>
            <div className="mt-3 h-72 overflow-auto rounded border border-slate-700 bg-slate-950 p-2 text-xs">
              {(rlStats?.history || []).map((row) => (
                <div key={row.id} className="mb-2 rounded border border-slate-800 p-2">
                  <div>reward: {Number(row.reward).toFixed(2)}</div>
                  <div>precision: {Number(row.precision).toFixed(3)}</div>
                  <div>threshold: {Number(row.threshold_before).toFixed(3)} → {Number(row.threshold_after).toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 lg:col-span-3">
            <h2 className="text-lg font-medium">Expert-in-the-Loop</h2>
            <div className="mt-3 overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="border-b border-slate-700 text-slate-300">
                  <tr>
                    <th className="px-3 py-2">Discovery</th>
                    <th className="px-3 py-2">Status</th>
                    <th className="px-3 py-2">Confidence</th>
                    <th className="px-3 py-2">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {discoveries.slice(0, 20).map((item) => (
                    <tr key={item.id} className="border-b border-slate-800">
                      <td className="px-3 py-2">{item.id}</td>
                      <td className="px-3 py-2">{item.status}</td>
                      <td className="px-3 py-2">{Number(item.confidence).toFixed(4)}</td>
                      <td className="px-3 py-2">
                        <div className="flex gap-2">
                          <button className="rounded bg-emerald-500 px-2 py-1 text-xs font-semibold text-slate-900" onClick={() => onFeedbackDiscovery(item.id, true)}>👍</button>
                          <button className="rounded bg-rose-500 px-2 py-1 text-xs font-semibold text-white" onClick={() => onFeedbackDiscovery(item.id, false)}>👎</button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        )}
        <div className="mt-4 text-xs text-slate-400">
          Uploaded points: {pointsCount} | Processed points: {normalizedPoints.length} | Time mode: {datasetMeta?.time_mode || "-"} | Range:{" "}
          {datasetMeta?.start_time || "-"} to {datasetMeta?.end_time || "-"}
        </div>
      </div>
      </div>
    </div>
  );

  if (screen === "home") {
    return renderHome();
  }
  if (screen === "login") {
    return renderLogin();
  }
  return renderDashboard();
}

export default App;
