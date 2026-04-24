import { useCallback, useEffect, useRef, useState } from 'react';
import useClickOutside from '../../hooks/useClickOutside';
import { Plus, Upload, FileIcon } from '../icons/Icons';

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
  const [files, setFiles] = useState<AttachedFile[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const inputId = 'attachment-file-input';

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

  const addFiles = (list: FileList) => {
    const next = Array.from(list).map((f) => ({
      id: `${f.name}-${f.size}-${Date.now()}`,
      name: f.name,
      size: f.size,
    }));
    setFiles((prev) => [...prev, ...next]);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
  };

  return (
    <div className="attach-wrap" ref={ref}>
      <button
        className="prompt-add"
        onClick={() => setOpen((o) => !o)}
        aria-label="Add attachment"
        aria-haspopup="true"
        aria-expanded={open}
      >
        <Plus size={16} strokeWidth={2} />
      </button>

      {open && (
        <div className="attach-menu" role="region" aria-label="Attachments">
          <label
            htmlFor={inputId}
            className={`attach-upload ${dragOver ? 'drag-over' : ''}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            <Upload />
            {dragOver ? 'Drop files here' : 'Upload Files'}
          </label>

          <input
            id={inputId}
            type="file"
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files?.length) addFiles(e.target.files);
              e.target.value = '';
            }}
          />

          <div className="attach-divider" />

          {files.length === 0 ? (
            <div className="attach-empty">
              No attachments yet. Drop files above or click to browse.
            </div>
          ) : (
            <div className="attach-list">
              {files.map((f) => (
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
