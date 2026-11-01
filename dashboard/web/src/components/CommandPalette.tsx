import { useEffect, useRef, useState } from "react";
import { useNavigate } from "@tanstack/react-router";

export type Command = { label: string; to: string; hash?: string };

/** ⌘K / Ctrl+K: filter commands by text, Enter jumps to the first match. Native <dialog> handles focus and Esc. */
export function CommandPalette({ commands }: { commands: Command[] }) {
  const ref = useRef<HTMLDialogElement>(null);
  const [q, setQ] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setQ("");
        ref.current?.showModal();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const hits = commands.filter((c) =>
    c.label.toLowerCase().includes(q.toLowerCase()),
  );
  const go = (c: Command) => {
    ref.current?.close();
    navigate({ to: c.to, hash: c.hash });
  };

  return (
    <dialog
      ref={ref}
      aria-label="Command palette"
      className="w-[32rem] max-w-[90vw] rounded border border-border bg-surface p-0 text-fg backdrop:bg-fg/30"
    >
      <input
        aria-label="Jump to"
        placeholder="Jump to a screen or gate"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && hits[0] && go(hits[0])}
        className="w-full border-b border-border bg-transparent px-3 py-2 outline-none"
      />
      <ul className="max-h-72 overflow-auto py-1">
        {hits.map((c) => (
          <li key={c.label}>
            <button
              onClick={() => go(c)}
              className="w-full px-3 py-1.5 text-left text-sm hover:bg-bg"
            >
              {c.label}
            </button>
          </li>
        ))}
        {hits.length === 0 && (
          <li className="px-3 py-1.5 text-sm text-muted">No match</li>
        )}
      </ul>
    </dialog>
  );
}
