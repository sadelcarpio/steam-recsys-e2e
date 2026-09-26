// App shell: header (user id box), routing and the two pages.
import "./style.css";
import {
  game,
  onlineRecommendations,
  userRecommendations,
  MAX_LIKED_GAMES,
  USER_ID_PATTERN,
  ApiError,
  type Recommendation,
  type RecommendationsResponse,
} from "./api";
import { cardGrid, detailsView, h, responseSummary } from "./cards";
import { t } from "./i18n";
import { LikedGames } from "./liked";
import { parseRoute, userPath } from "./router";
import type { GameHit } from "./search";

const main = document.querySelector<HTMLElement>("#main")!;
const dialog = document.querySelector<HTMLDialogElement>("#details")!;
const liked = new LikedGames();

// ---- navigation --------------------------------------------------------------------------

function navigate(path: string): void {
  if (path !== location.pathname) history.pushState(null, "", path);
  render();
}

window.addEventListener("popstate", render);
document.addEventListener("click", (e) => {
  const link = (e.target as Element).closest<HTMLAnchorElement>("a[data-nav]");
  if (link && !e.metaKey && !e.ctrlKey) {
    e.preventDefault();
    navigate(link.pathname);
  }
});

document.querySelector<HTMLFormElement>("#user-form")!.addEventListener("submit", (e) => {
  e.preventDefault();
  const input = (e.currentTarget as HTMLFormElement).elements.namedItem("user") as HTMLInputElement;
  const userId = input.value.trim();
  if (!USER_ID_PATTERN.test(userId)) {
    input.setCustomValidity(t.userIdInvalid);
    input.reportValidity();
    return;
  }
  input.setCustomValidity("");
  navigate(userPath(userId));
});

function render(): void {
  const route = parseRoute(location.pathname);
  main.replaceChildren();
  if (route.page === "discover") discoverPage();
  else if (route.page === "user") userPage(route.userId);
  else
    main.append(
      h("p", { class: "empty" }, t.notFound, h("a", { href: "/", "data-nav": "" }, t.goHome)),
    );
}

// ---- shared ------------------------------------------------------------------------------

async function openDetails(rec: Recommendation): Promise<void> {
  dialog.replaceChildren(h("p", { class: "status" }, t.loading));
  dialog.showModal();
  try {
    const details = rec.details ?? (await game(rec.game_id));
    dialog.replaceChildren(detailsView(details, rec.explanation), closeButton());
  } catch (err) {
    dialog.replaceChildren(h("p", { class: "error" }, errorText(err)), closeButton());
  }
}

function closeButton(): HTMLElement {
  const button = h("button", { class: "close", "aria-label": t.close }, "×");
  button.addEventListener("click", () => dialog.close());
  return button;
}

dialog.addEventListener("click", (e) => {
  if (e.target === dialog) dialog.close(); // backdrop
});

function errorText(err: unknown): string {
  if (err instanceof ApiError && err.detail) console.warn(err.status, err.detail);
  if (!(err instanceof ApiError)) console.warn(err);
  return err instanceof ApiError ? err.message : t.errors.unknown;
}

/** Renders `load()` into `section`, replacing what was there. */
async function showRecommendations(
  section: HTMLElement,
  load: () => Promise<RecommendationsResponse>,
  extra?: (r: RecommendationsResponse) => Node | null,
): Promise<void> {
  section.replaceChildren(h("p", { class: "status" }, t.loadingRecommendations));
  try {
    const response = await load();
    section.replaceChildren(
      h("p", { class: "summary" }, responseSummary(response)),
      extra?.(response) ?? "",
      cardGrid(response, openDetails),
    );
  } catch (err) {
    section.replaceChildren(h("p", { class: "error" }, errorText(err)));
  }
}

// ---- /u/<id> -----------------------------------------------------------------------------

function userPage(userId: string): void {
  const section = h("section", {});
  main.append(h("h1", {}, t.userTitle, h("code", {}, userId)), section);
  (document.querySelector("#user-form input") as HTMLInputElement).value = userId;
  void showRecommendations(section, () => userRecommendations(userId));
}

// ---- / (discover) ------------------------------------------------------------------------

let worker: Worker | undefined;
let searchReady: Promise<number> | undefined;
let searchSeq = 0;
const pending = new Map<number, (hits: GameHit[]) => void>();

function startSearch(): Promise<number> {
  if (searchReady) return searchReady;
  worker = new Worker(new URL("./search.worker.ts", import.meta.url), { type: "module" });
  searchReady = new Promise((resolve, reject) => {
    worker!.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === "ready") resolve(msg.size);
      else if (msg.type === "error") reject(new Error(msg.message));
      else if (msg.type === "results") {
        pending.get(msg.id)?.(msg.hits);
        pending.delete(msg.id);
      }
    };
  });
  return searchReady;
}

function search(query: string): Promise<GameHit[]> {
  const id = ++searchSeq;
  return new Promise((resolve) => {
    pending.set(id, resolve);
    worker!.postMessage({ id, query });
  });
}

function discoverPage(): void {
  const input = h("input", {
    type: "search",
    placeholder: t.searchLoading,
    "aria-label": t.searchLabel,
    autocomplete: "off",
    disabled: "",
  });
  const results = h("ul", { class: "results", role: "listbox" });
  const chips = h("ul", { class: "chips" });
  const count = h("p", { class: "hint count" });
  const recommend = h("button", { class: "primary" }, t.recommend);
  const clear = h("button", {}, t.clear);
  const section = h("section", {});
  let searchSize: number | undefined; // set once the index is loaded

  main.append(
    h("h1", {}, t.discoverTitle),
    h("p", { class: "lead" }, t.discoverLead(MAX_LIKED_GAMES)),
    h("div", { class: "search" }, input, results),
    h("div", { class: "liked" }, chips, h("div", { class: "actions" }, recommend, clear)),
    count,
    section,
  );

  // The search box is closed while the list is full (and until the index is loaded).
  const updateInput = () => {
    if (searchSize === undefined) return;
    input.toggleAttribute("disabled", liked.full);
    input.placeholder = liked.full
      ? t.searchFull(MAX_LIKED_GAMES)
      : t.searchPlaceholder(searchSize);
  };

  const renderLiked = () => {
    chips.replaceChildren(
      ...liked.games.map((g) => {
        const remove = h("button", { "aria-label": t.remove(g.name) }, "×");
        remove.addEventListener("click", () => {
          liked.remove(g.game_id);
          renderLiked();
        });
        return h("li", {}, g.name, remove);
      }),
    );
    if (!liked.games.length) chips.append(h("li", { class: "hint" }, t.noneChosen));
    count.textContent = t.picked(liked.games.length, MAX_LIKED_GAMES);
    recommend.toggleAttribute("disabled", !liked.games.length);
    clear.toggleAttribute("disabled", !liked.games.length);
    updateInput();
  };

  const pick = (hit: GameHit) => {
    if (!liked.add(hit)) return; // full
    renderLiked();
    input.value = "";
    results.replaceChildren();
    if (!liked.full) input.focus();
  };

  let latest = 0;
  input.addEventListener("input", async () => {
    const seq = ++latest;
    const query = input.value;
    const hits = await search(query);
    if (seq !== latest) return; // a newer query answered first
    if (query.trim() && !hits.length) {
      results.replaceChildren(h("li", { class: "hint" }, t.noResults));
      return;
    }
    results.replaceChildren(
      ...hits.map((hit) => {
        const item = h("li", { role: "option", tabindex: "0" }, h("span", {}, hit.name));
        item.addEventListener("click", () => pick(hit));
        item.addEventListener("keydown", (e) => {
          if (e.key === "Enter") pick(hit);
        });
        return item;
      }),
    );
  });

  const startHint = () => section.replaceChildren(h("p", { class: "empty" }, t.startHint));

  recommend.addEventListener("click", () =>
    showRecommendations(
      section,
      () => onlineRecommendations(liked.games.map((g) => g.game_id)),
      (r) =>
        r.ignored_game_ids?.length
          ? h("p", { class: "hint" }, t.unknownPicked(r.ignored_game_ids.length))
          : null,
    ),
  );
  clear.addEventListener("click", () => {
    liked.clear();
    renderLiked();
    startHint();
  });

  renderLiked();
  startHint();
  startSearch().then(
    (size) => {
      searchSize = size;
      updateInput();
    },
    (err) => {
      console.warn(err);
      input.placeholder = t.searchUnavailable;
    },
  );
}

render();
