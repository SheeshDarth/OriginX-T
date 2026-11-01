import { useEffect, useRef } from "react";

/** The fixed contamination palette (HANDBOOK §8.6), same as reports/figures and tokens.css. */
export const TYPE_COLORS: Record<string, string> = {
  synthetic: "#2a78d6",
  recursive: "#eb6834",
  paraphrased: "#1baf7a",
  benchmark_near: "#eda100",
  human_control: "#8a8a8a",
};
export const TYPES = Object.keys(TYPE_COLORS);

/** Mounts an Observable Plot figure; `make` rebuilds it whenever `deps` change. */
export function PlotFigure({
  make,
  deps,
  caption,
}: {
  make: () => SVGSVGElement | HTMLElement;
  deps: unknown[];
  caption: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = make();
    el.setAttribute("aria-hidden", "true"); // Plot labels its <g>s without roles; the figure speaks as one image
    ref.current?.replaceChildren(el);
    return () => el.remove();
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <figure>
      <div ref={ref} role="img" aria-label={caption} />
      <figcaption className="mt-1 text-xs text-muted">{caption}</figcaption>
    </figure>
  );
}
