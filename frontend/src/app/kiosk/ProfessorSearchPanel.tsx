"use client";

import { ExternalLink, LoaderCircle, Search, UsersRound } from "lucide-react";
import type { ProfessorSearchState } from "./useRealtimeSession";

export function ProfessorSearchPanel({
  search,
}: {
  search: ProfessorSearchState | null;
}) {
  if (!search || search.status === "cancelled") return null;

  const searching = search.status === "searching";
  const failed = search.status === "error";

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
          {search.query && <p className="k-professor-query">“{search.query}”</p>}
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
