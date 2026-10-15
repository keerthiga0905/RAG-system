import React, { useEffect, useState } from 'react';
import { Database, ExternalLink, Search, FileText, AlertTriangle } from 'lucide-react';
import type { DocumentSummary } from '../types';

export const DatasetInventory: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [filter, setFilter] = useState<string>('');
  const [categoryFilter, setCategoryFilter] = useState<string>('All');

  useEffect(() => {
    fetch('http://localhost:8000/api/documents')
      .then(res => res.json())
      .then((data: DocumentSummary[]) => {
        setDocuments(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const categories: string[] = ['All', ...Array.from(new Set(documents.map((d: DocumentSummary) => d.category)))];

  const filteredDocs = documents.filter((d: DocumentSummary) => {
    const matchesText = d.title.toLowerCase().includes(filter.toLowerCase()) || 
                        d.document_id.toLowerCase().includes(filter.toLowerCase());
    const matchesCat = categoryFilter === 'All' || d.category === categoryFilter;
    return matchesText && matchesCat;
  });

  const totalChunks = documents.reduce((acc: number, d: DocumentSummary) => acc + d.chunks_count, 0);

  return (
    <div className="animate-fade-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Database color="var(--color-primary)" />
            <span>Official CIT Dataset Inventory &amp; Corpus Registry</span>
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
            14 verified official documents (10 primary datasets + 4 supplementary brochures/regulations PDFs) preserved in <code>data/raw/</code>.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className="badge-pill blue">14 Discovered Documents</span>
          <span className="badge-pill green">{totalChunks} Ingested Chunks</span>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="table-card" style={{ padding: '1rem 1.25rem', marginBottom: '1.5rem', display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <div style={{ flex: 1, minWidth: '240px', display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--bg-main)', padding: '0.4rem 0.8rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          <Search size={16} color="var(--text-light)" />
          <input
            type="text"
            placeholder="Search documents by title or filename..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', fontSize: '0.875rem' }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>Category:</span>
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="category-select"
          >
            {categories.map((c: string) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>

      {/* Documents Table */}
      <div className="table-card">
        {loading ? (
          <div style={{ textAlign: 'center', padding: '2rem' }}>Loading documents...</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Document File</th>
                <th>Document Title</th>
                <th>Category</th>
                <th>Format</th>
                <th>Chunks</th>
                <th>Academic Year</th>
                <th>Official Source</th>
              </tr>
            </thead>
            <tbody>
              {filteredDocs.map((doc: DocumentSummary, idx: number) => (
                <tr key={doc.document_id}>
                  <td>{idx + 1}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600, fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                      <FileText size={14} color="var(--color-primary)" />
                      <span>{doc.document_id}</span>
                    </div>
                  </td>
                  <td style={{ fontWeight: 600 }}>{doc.title}</td>
                  <td>
                    <span className="badge-pill amber">{doc.category}</span>
                  </td>
                  <td>
                    <span className={`badge-pill ${doc.format === 'PDF' ? 'red' : 'blue'}`}>{doc.format}</span>
                  </td>
                  <td style={{ fontWeight: 700 }}>{doc.chunks_count}</td>
                  <td>{doc.academic_year || 'N/A'}</td>
                  <td>
                    <a
                      href={doc.source_url}
                      target="_blank"
                      rel="noreferrer"
                      style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.775rem', fontWeight: 600 }}
                    >
                      <span>Link</span>
                      <ExternalLink size={12} />
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Official Conflict Governance Card */}
      <div className="table-card">
        <h3 style={{ fontSize: '1.15rem', color: 'var(--color-primary)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <AlertTriangle size={18} />
          <span>Tracked Official Data Conflicts &amp; Resolution Rules</span>
        </h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '1rem' }}>
          CIT official publications contain a small number of documented conflicting claims. In accordance with honest ML evaluation standards, these are tracked and surfaced directly to prevent hallucination:
        </p>

        <ul style={{ paddingLeft: '1.5rem', color: 'var(--text-main)', fontSize: '0.875rem', lineHeight: '1.8' }}>
          <li>
            <strong>CONF-001 (Autonomous Status Date):</strong> AQAR 2022-2023 states autonomous status was conferred on <code>01/06/1987</code>, while AQAR 2023-2024 states <code>08/07/1987</code>. The RAG engine cites both official reporting years.
          </li>
          <li>
            <strong>CONF-002 (2022-2023 Placed Students):</strong> 2022-2023 Placement summary table records <code>757</code> placed students, while the 6-year trend table records <code>782</code> placed students due to differing eligibility denominator rules (no active backlogs vs interested students).
          </li>
          <li>
            <strong>CONF-003 (Principal Contact):</strong> The official Contact Us page lists <code>94868 37757 / principal.citoffice@cit.edu.in</code>, whereas the Information for Students page lists <code>+91-422-2573187 / principal@cit.edu.in</code>. Both channels are surfaced.
          </li>
          <li>
            <strong>CONF-004 (B.E. VLSI Design &amp; Technology):</strong> The 2026 brochure lists the programme as offered since 2026, marked with an asterisk (<code>* awaiting approval</code>). The RAG system explicitly warns the user of the pending regulatory approval status.
          </li>
        </ul>
      </div>
    </div>
  );
};
