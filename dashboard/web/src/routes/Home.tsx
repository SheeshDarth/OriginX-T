import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import * as Plot from "@observablehq/plot";
import { api, unwrap, type Curve, type Gate } from "../api/client";
import { PlotFigure, TYPE_COLORS, fit } from "../components/charts";
import { StatusBadge } from "../components/StatusBadge";
import { evidence } from "./Overview";

const SHOWN = ["recursive", "synthetic", "paraphrased", "human_control"];

/** The WikiText-2 dose-response, drawn from the committed grid (no placeholder data). */
export function heroPlot(curves: Curve[]) {
  const rows = curves.filter((c) => SHOWN.includes(c.type));
  const plot = Plot.plot({
    width: 560,
    height: 360,
    marginLeft: 48,
    marginRight: 92,
    marginBottom: 52,
    style: {
      background: "transparent",
      color: "currentColor",
      fontSize: "12px",
    },
    x: {
      label: "Share of the training set that is contaminated",
      labelAnchor: "center",
      ticks: [0, 0.25, 0.5, 0.75, 1],
      tickFormat: (d: number) => `${d * 100}%`,
    },
    y: { label: "Hidden-holdout perplexity (lower is better)", grid: true },
    marks: [
      Plot.lineY(rows, {
        x: "ratio",
        y: "holdout_ppl_mean",
        z: "type",
        stroke: (c: Curve) => TYPE_COLORS[c.type],
        strokeWidth: 2.5,
      }),
      Plot.text(
        rows.filter((c) => c.ratio === 1),
        {
          x: "ratio",
          y: "holdout_ppl_mean",
          text: (c: Curve) => c.type.replace("_", " "),
          fill: (c: Curve) => TYPE_COLORS[c.type],
          textAnchor: "start",
          dx: 8,
        },
      ),
    ],
  });
  // draw each line in on load (CSS .anim-draw reads pathLength)
  plot
    .querySelectorAll('[aria-label="line"] path')
    .forEach((p) => p.setAttribute("pathLength", "1"));
  return fit(plot, 560, 360);
}

const STEPS = [
  {
    n: "01",
    title: "Upload",
    text: "A JSONL file of training records. Nothing leaves your machine unless you point the app at a remote API.",
  },
  {
    n: "02",
    title: "Measure",
    text: "Lexical statistics, benchmark similarity, a reference language model and sentence embeddings, 22 features in all.",
  },
  {
    n: "03",
    title: "Predict",
    text: "A risk model turns the features into a probability of degradation, with a conformal set that can say “uncertain”.",
  },
  {
    n: "04",
    title: "Explain",
    text: "The features that drive the call, and the individual samples to remove first.",
  },
];

function Evidence({ gates }: { gates: Gate[] }) {
  const passed = gates.filter((g) => g.status === "passed").length;
  const failed = gates.filter((g) => g.status === "failed").length;
  return (
    <section aria-labelledby="evidence-title" className="mt-24">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <h2
          id="evidence-title"
          className="text-lg font-semibold tracking-tight"
        >
          What we tested, and what failed
        </h2>
        <span className="font-mono text-xs text-muted">
          {passed} passed · {failed} failed · every rule fixed before its result
        </span>
      </div>
      <div className="overflow-x-auto rounded border border-border bg-surface">
        <table className="w-full text-sm">
          <caption className="sr-only">
            Pre-registered gates and their verdicts
          </caption>
          <tbody>
            {gates.map((g) => (
              <tr
                key={`${g.id}-${g.dataset ?? ""}`}
                className="border-b border-border last:border-0"
              >
                <th
                  scope="row"
                  className="whitespace-nowrap px-4 py-2 text-left font-medium"
                >
                  {g.name}
                  {g.dataset && (
                    <span className="ml-2 font-mono text-xs font-normal text-muted">
                      {g.dataset}
                    </span>
                  )}
                </th>
                <td className="px-4 py-2">
                  <StatusBadge status={g.status} />
                </td>
                <td className="px-4 py-2 text-xs text-muted">{evidence(g)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-muted">
        <Link to="/overview" className="text-accent underline">
          Open the full gate board
        </Link>{" "}
        for each rule, its numbers and the pull request behind it.
      </p>
    </section>
  );
}

export function Home() {
  const curves = useQuery({
    queryKey: ["curves", "wikitext2"],
    queryFn: () =>
      unwrap(
        api.GET("/v1/grids/{dataset}/curves", {
          params: { path: { dataset: "wikitext2" } },
        }),
      ),
  });
  const gates = useQuery({
    queryKey: ["gates"],
    queryFn: () => unwrap(api.GET("/v1/gates")),
  });
  return (
    <div>
      <section className="grid grid-cols-[minmax(0,1fr)] items-center gap-12 pt-6 lg:grid-cols-[minmax(0,1fr)_35rem] lg:pt-14">
        <div>
          <p className="anim-fade-up font-mono text-xs uppercase tracking-widest text-muted">
            ORIGIN-T · dataset forensics for LLM fine-tuning
          </p>
          <h1
            className="anim-fade-up mt-4 text-4xl font-semibold leading-[1.08] tracking-tight lg:text-5xl"
            style={{ animationDelay: "60ms" }}
          >
            Know whether a dataset will damage your model before you train on
            it.
          </h1>
          <p
            className="anim-fade-up mt-5 max-w-xl text-base leading-relaxed text-muted"
            style={{ animationDelay: "120ms" }}
          >
            ORIGIN-T audits a fine-tuning dataset for generated, recursive and
            leaked text. It returns a calibrated risk of degradation, the
            features behind it, and the samples to blame.
          </p>
          <div
            className="anim-fade-up mt-8 flex flex-wrap items-center gap-3"
            style={{ animationDelay: "180ms" }}
          >
            <Link
              to="/try"
              className="rounded bg-accent px-5 py-2.5 text-sm font-medium text-bg transition-opacity hover:opacity-90"
            >
              Try it now
            </Link>
            <Link
              to="/overview"
              className="rounded border border-border px-5 py-2.5 text-sm font-medium transition-colors hover:border-accent/60"
            >
              See the evidence
            </Link>
          </div>
          <p
            className="anim-fade-up mt-6 max-w-xl text-xs text-muted"
            style={{ animationDelay: "240ms" }}
          >
            A research prototype, validated on controlled DistilGPT-2 + LoRA
            experiments. A screen for datasets, not a guarantee.
          </p>
        </div>
        <div className="anim-draw min-w-0">
          {curves.data ? (
            <PlotFigure
              make={() => heroPlot(curves.data!)}
              deps={[curves.data]}
              caption="Mixing in more generated text raises perplexity on held-out human text; the human-text control stays flat. DistilGPT-2 + LoRA, WikiText-2, mean of 3 seeds."
            />
          ) : (
            <div
              aria-hidden="true"
              className="h-[360px] rounded border border-border bg-surface/50"
            />
          )}
        </div>
      </section>

      <section aria-labelledby="how-title" className="mt-24">
        <h2
          id="how-title"
          className="mb-6 text-lg font-semibold tracking-tight"
        >
          How an audit works
        </h2>
        <ol className="grid gap-px overflow-hidden rounded border border-border bg-border sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((s) => (
            <li key={s.n} className="bg-surface p-5">
              <span className="font-mono text-xs text-accent">{s.n}</span>
              <h3 className="mt-2 text-sm font-semibold">{s.title}</h3>
              <p className="mt-1 text-sm leading-relaxed text-muted">
                {s.text}
              </p>
            </li>
          ))}
        </ol>
        <p className="mt-4">
          <Link to="/try" className="text-sm text-accent underline">
            Run one on a sample dataset
          </Link>
        </p>
      </section>

      {gates.data && <Evidence gates={gates.data} />}

      <footer className="mt-24 border-t border-border pt-6 text-xs text-muted">
        ORIGIN-T is a final-year research project (23AML175). Source and
        pre-registered rules:{" "}
        <a
          href="https://github.com/SheeshDarth/OriginX-T"
          className="text-accent underline"
        >
          github.com/SheeshDarth/OriginX-T
        </a>
        .
      </footer>
    </div>
  );
}
