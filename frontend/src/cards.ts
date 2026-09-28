// Rendering: game cards, the details dialog, and a tiny element helper. Text always goes through
// textContent (names, descriptions and LLM explanations are data, never HTML).
import {
  headerImage,
  type GameDetails,
  type Recommendation,
  type RecommendationsResponse,
} from "./api";
import { formatDate, genreName, t } from "./i18n";

type Child = Node | string | null | undefined | false;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Record<string, string> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) el.setAttribute(key, value);
  for (const child of children) {
    if (child) el.append(child);
  }
  return el;
}

export function storeUrl(gameId: number): string {
  return `https://store.steampowered.com/app/${gameId}/`;
}

export function priceLabel(details?: GameDetails): string | null {
  if (!details) return null;
  if (details.is_free) return t.free;
  if (details.price == null) return null;
  return `$${details.price.toFixed(2)}`;
}

function image(game: { game_id: number; name: string; details?: GameDetails }): HTMLImageElement {
  const img = h("img", { src: headerImage(game), alt: game.name, loading: "lazy" });
  img.addEventListener("error", () => img.classList.add("missing"), { once: true });
  return img;
}

function tags(details?: GameDetails): HTMLElement | null {
  const genres = details?.genres?.slice(0, 3) ?? [];
  const price = priceLabel(details);
  if (!genres.length && !price) return null;
  return h(
    "ul",
    { class: "tags" },
    ...genres.map((g) => h("li", {}, genreName(g))),
    price && h("li", { class: "price" }, price),
  );
}

export function gameCard(rec: Recommendation, onOpen: (rec: Recommendation) => void): HTMLElement {
  const card = h(
    "article",
    { class: "card", tabindex: "0", "data-game-id": String(rec.game_id) },
    image(rec),
    h(
      "div",
      { class: "card-body" },
      h("h3", {}, h("span", { class: "rank" }, `#${rec.rank}`), " ", rec.name),
      tags(rec.details),
      rec.explanation && h("p", { class: "explanation" }, rec.explanation),
    ),
  );
  card.addEventListener("click", () => onOpen(rec));
  card.addEventListener("keydown", (e) => {
    if (e.key === "Enter") onOpen(rec);
  });
  return card;
}

export function cardGrid(
  response: RecommendationsResponse,
  onOpen: (rec: Recommendation) => void,
): HTMLElement {
  if (!response.recommendations.length) return h("p", { class: "empty" }, t.noRecommendations);
  return h("div", { class: "grid" }, ...response.recommendations.map((r) => gameCard(r, onOpen)));
}

export function responseSummary(response: RecommendationsResponse): string {
  const when = formatDate(response.generated_at);
  switch (response.source) {
    case "personalized":
      return response.reranked ? t.summaryReranked(when) : t.summaryPersonalized(when);
    case "online":
      return t.summaryOnline(response.model_id);
    case "popular":
      return response.user_id ? t.summaryFallback(when) : t.summaryPopular(when);
  }
}

export function detailsView(details: GameDetails, explanation?: string): HTMLElement {
  const facts: [string, string | undefined][] = [
    [t.released, details.release_date],
    [t.price, priceLabel(details) ?? undefined],
    [t.developer, details.developers?.join(", ")],
    [t.publisher, details.publishers?.join(", ")],
    [t.genres, details.genres?.map(genreName).join(", ")],
  ];
  return h(
    "div",
    { class: "details" },
    image(details),
    h("h2", {}, details.name),
    explanation && h("p", { class: "explanation" }, explanation),
    details.short_description && h("p", {}, details.short_description),
    h(
      "dl",
      {},
      ...facts.filter(([, v]) => v).flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v as string)]),
    ),
    h("a", { href: storeUrl(details.game_id), target: "_blank", rel: "noopener" }, t.viewOnSteam),
  );
}
