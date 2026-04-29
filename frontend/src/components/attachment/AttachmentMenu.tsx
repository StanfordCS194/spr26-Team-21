import { useCallback, useEffect, useRef, useState } from 'react';
import useClickOutside from '../../hooks/useClickOutside';
import { Plus, Upload, FileIcon, Check } from '../icons/Icons';

interface AttachedFile {
  id: string;
  name: string;
  size: number;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function AttachmentMenu() {
  const [open, setOpen] = useState(false);
  const [groundingFiles, setGroundingFiles] = useState<AttachedFile[]>([]);
  const [contextFiles, setContextFiles] = useState<AttachedFile[]>([]);
  const [dragOverGrounding, setDragOverGrounding] = useState(false);
  const [dragOverContext, setDragOverContext] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const groundingInputId = 'attachment-grounding-input';
  const contextInputId = 'attachment-context-input';

  const close = useCallback(() => setOpen(false), []);
  useClickOutside(ref, close);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [open]);

  const makeFiles = (list: FileList): AttachedFile[] =>
    Array.from(list).map((f) => ({
      id: `${f.name}-${f.size}-${Date.now()}`,
      name: f.name,
      size: f.size,
    }));

  const totalAttached = groundingFiles.length + contextFiles.length;

  return (
    <div className="attach-wrap" ref={ref}>
      <button
        className={`prompt-add${totalAttached > 0 ? ' prompt-add-active' : ''}`}
        onClick={() => setOpen((o) => !o)}
        aria-label="Add attachment"
        aria-haspopup="true"
        aria-expanded={open}
      >
        <Plus size={16} strokeWidth={2} />
        {totalAttached > 0 && (
          <span className="prompt-add-badge">{totalAttached}</span>
        )}
      </button>

      {open && (
        <div className="attach-menu" role="region" aria-label="Attachments">

          {/* — Grounding section — */}
          <div className="attach-section-label">Ground in real data</div>
          <label
            htmlFor={groundingInputId}
            className={`attach-upload attach-upload-grounding${dragOverGrounding ? ' drag-over' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOverGrounding(true); }}
            onDragLeave={() => setDragOverGrounding(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOverGrounding(false);
              if (e.dataTransfer.files?.length)
                setGroundingFiles((p) => [...p, ...makeFiles(e.dataTransfer.files)]);
            }}
          >
            <Upload size={14} />
            {dragOverGrounding ? 'Drop to ground generation' : 'Drop CSV · XLSX · Parquet'}
          </label>
          <input
            id={groundingInputId}
            type="file"
            multiple
            hidden
            accept=".csv,.xlsx,.xls,.parquet"
            onChange={(e) => {
              if (e.target.files?.length)
                setGroundingFiles((p) => [...p, ...makeFiles(e.target.files!)]);
              e.target.value = '';
            }}
          />

          {groundingFiles.length > 0 && (
            <div className="attach-list">
              {groundingFiles.map((f) => (
                <div key={f.id} className="attach-row attach-row-grounding">
                  <FileIcon />
                  <span className="attach-name">{f.name}</span>
                  <span className="attach-grounding-badge">
                    <Check size={9} strokeWidth={2.5} />
                    Grounding
                  </span>
                  <span className="attach-size">{formatSize(f.size)}</span>
                </div>
              ))}
            </div>
          )}

          <div className="attach-divider" />

          {/* — Additional context section — */}
          <div className="attach-section-label">Additional context</div>
          <label
            htmlFor={contextInputId}
            className={`attach-upload${dragOverContext ? ' drag-over' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOverContext(true); }}
            onDragLeave={() => setDragOverContext(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOverContext(false);
              if (e.dataTransfer.files?.length)
                setContextFiles((p) => [...p, ...makeFiles(e.dataTransfer.files)]);
            }}
          >
            <Upload size={14} />
            {dragOverContext ? 'Drop files here' : 'Upload files'}
          </label>
          <input
            id={contextInputId}
            type="file"
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files?.length)
                setContextFiles((p) => [...p, ...makeFiles(e.target.files!)]);
              e.target.value = '';
            }}
          />

          {contextFiles.length === 0 && groundingFiles.length === 0 && (
            <div className="attach-empty">
              Upload your real data above to ground synthetic generation.
            </div>
          )}

          {contextFiles.length > 0 && (
            <div className="attach-list">
              {contextFiles.map((f) => (
                <div key={f.id} className="attach-row">
                  <FileIcon />
                  <span className="attach-name">{f.name}</span>
                  <span className="attach-size">{formatSize(f.size)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
