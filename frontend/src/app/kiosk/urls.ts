// URL helpers for the kiosk.
//
// Dev (local): orchestrator on :8001, CV pipeline on :8000 (override with
//   NEXT_PUBLIC_ORCH_PORT / NEXT_PUBLIC_BACKEND_PORT).
// Docker: reverse-proxied under /orch (orchestrator) and /api (CV pipeline),
//   same host, scheme derived from the page (ws/wss).

function inDocker(): boolean {
  return ["true", "1"].includes(
    (process.env.NEXT_PUBLIC_IN_DOCKER || "").toLowerCase()
  );
}

function wsProto(): string {
  return window.location.protocol === "https:" ? "wss:" : "ws:";
}

export function getOrchestratorWsUrl(sessionId: string): string {
  const q = `?session_id=${encodeURIComponent(sessionId)}`;
  if (inDocker()) {
    return `${wsProto()}//${window.location.host}/orch/v1/realtime${q}`;
  }
  const port = process.env.NEXT_PUBLIC_ORCH_PORT || "8001";
  return `${wsProto()}//${window.location.hostname}:${port}/v1/realtime${q}`;
}

export function getCvWsUrl(channel: string, sessionId: string): string {
  const path = `${channel}/${encodeURIComponent(sessionId)}`;
  if (inDocker()) {
    return `${wsProto()}//${window.location.host}/api/${path}`;
  }
  const port = process.env.NEXT_PUBLIC_BACKEND_PORT || "8000";
  return `${wsProto()}//${window.location.hostname}:${port}/${path}`;
}
