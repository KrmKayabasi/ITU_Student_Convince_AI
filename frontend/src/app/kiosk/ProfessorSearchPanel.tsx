"use client";

import { useEffect, useRef, useState } from "react";
import { ExternalLink, LoaderCircle, QrCode, Search, UsersRound, X } from "lucide-react";
import type { ProfessorSearchState } from "./useRealtimeSession";

/** Seconds before the result panel auto-collapses into a QR code. */
const AUTO_COLLAPSE_MS = 3000;

export function ProfessorSearchPanel({
  search,
}: {
  search: ProfessorSearchState | null;
}) {
  const [showQR, setShowQR] = useState(false);
  const collapseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Reset QR state whenever a new search arrives.
  useEffect(() => {
    setShowQR(false);
    if (collapseTimerRef.current !== null) {
      clearTimeout(collapseTimerRef.current);
      collapseTimerRef.current = null;
    }
  }, [search?.id, search?.status]);

  // Auto-collapse: after results have been visible for 3 seconds, show QR code.
  useEffect(() => {
    if (!search || search.status !== "completed" || search.results.length === 0) {
      return;
    }
    collapseTimerRef.current = setTimeout(() => {
      setShowQR(true);
    }, AUTO_COLLAPSE_MS);

    return () => {
      if (collapseTimerRef.current !== null) {
        clearTimeout(collapseTimerRef.current);
        collapseTimerRef.current = null;
      }
    };
  }, [search, search?.id, search?.status, search?.results?.length]);

  if (!search || search.status === "cancelled") return null;

  const searching = search.status === "searching";
  const failed = search.status === "error";

  // ── QR Code view (after auto-collapse) ──────────────────────────────────
  if (showQR && search.results.length > 0 && search.sourceUrl) {
    const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(search.sourceUrl)}`;
    return (
      <aside className="k-professor-panel k-professor-panel--qr" aria-live="polite">
        <button
          className="k-professor-qr-close"
          onClick={() => setShowQR(false)}
          aria-label="Profesör listesini tekrar göster"
        >
          <X size={18} />
        </button>
        <div className="k-professor-qr-body">
          <QrCode size={24} aria-hidden />
          <p className="k-professor-qr-title">Akademisyenleri telefonuna aktar</p>
          <p className="k-professor-qr-hint">
            QR kodu tara — {search.results.length} akademisyenin profili cebinde.
          </p>
          <img
            className="k-professor-qr-image"
            src={qrUrl}
            alt={`İTÜ Akademi aramasının QR kodu: ${search.query}`}
            width={160}
            height={160}
          />
          <p className="k-professor-qr-query">&ldquo;{search.query}&rdquo;</p>
          <a
            className="k-professor-source"
            href={search.sourceUrl}
            target="_blank"
            rel="noreferrer"
          >
            <UsersRound size={15} aria-hidden />
            Kaynak: {search.sourceName ?? "İTÜ Akademi"}
            <ExternalLink size={13} aria-hidden />
          </a>
        </div>
      </aside>
    );
  }

  // ── Normal result / searching / error view ──────────────────────────────
  return (
    <aside className="k-professor-panel" aria-live="polite">
      <div className="k-professor-heading">
        <span className="k-professor-icon" aria-hidden>
          {searching ? (
            <LoaderCircle className="k-search-spin" size={20} />
          ) : (
            <Search size={20} />
          )}
        </span>
        <div className="min-w-0">
          <p className="k-professor-eyebrow">İTÜ Akademi</p>
          <h2>{searching ? "Akademisyenler aranıyor" : "Akademisyen eşleşmeleri"}</h2>
          {search.query && <p className="k-professor-query">&ldquo;{search.query}&rdquo;</p>}
        </div>
      </div>

      {searching && (
        <div className="k-search-progress">
          <span />
          <p>Resmî profiller ve çalışma alanları taranıyor.</p>
        </div>
      )}

      {failed && (
        <p className="k-search-empty">
          {search.message ?? "Arama şu anda tamamlanamadı."}
        </p>
      )}

      {!searching && !failed && search.results.length === 0 && (
        <p className="k-search-empty">
          Resmî İTÜ Akademi kayıtlarında bu konuyla eşleşen profil bulunamadı.
        </p>
      )}

      {search.results.length > 0 && (
        <div className="k-professor-results">
          {search.results.map((professor) => (
            <a
              className="k-professor-card"
              href={professor.profile_url}
              target="_blank"
              rel="noreferrer"
              key={professor.profile_url}
            >
              <span className="k-professor-avatar" aria-hidden>
                {professor.name
                  .split(" ")
                  .slice(0, 2)
                  .map((part) => part[0])
                  .join("")}
              </span>
              <span className="min-w-0 flex-1">
                <strong>{professor.title} {professor.name}</strong>
                {professor.department && <small>{professor.department}</small>}
                {professor.work_areas && <em>{professor.work_areas}</em>}
              </span>
              <ExternalLink className="shrink-0" size={16} aria-hidden />
            </a>
          ))}
        </div>
      )}

      {!searching && search.sourceUrl && (
        <a
          className="k-professor-source"
          href={search.sourceUrl}
          target="_blank"
          rel="noreferrer"
        >
          <UsersRound size={15} aria-hidden />
          Kaynak: {search.sourceName ?? "İTÜ Akademi"}
          <ExternalLink size={13} aria-hidden />
        </a>
      )}
    </aside>
  );
}
