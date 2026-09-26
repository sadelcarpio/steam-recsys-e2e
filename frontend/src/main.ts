// App shell: header (user id box), routing and the two pages.
import "./style.css";
import {
  game,
  onlineRecommendations,
  popular,
  userRecommendations,
  MAX_LIKED_GAMES,
  USER_ID_PATTERN,
  type Recommendation,
  type RecommendationsResponse,
} from "./api";
import { cardGrid, detailsView, h, responseSummary } from "./cards";
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
    input.setCustomValidity("A Steam account id: 1 to 20 digits");
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
      h(
        "p",
        { class: "empty" },
        "Page not found. ",
        h("a", { href: "/", "data-nav": "" }, "Go home"),
      ),
    );
}

// ---- shared ------------------------------------------------------------------------------

async function openDetails(rec: Recommendation): Promise<void> {
  dialog.replaceChildren(h("p", { class: "status" }, "Loading…"));
  dialog.showModal();
  try {
    const details = rec.details ?? (await game(rec.game_id));
    dialog.replaceChildren(detailsView(details, rec.explanation), closeButton());
  } catch (err) {
    dialog.replaceChildren(h("p", { class: "error" }, errorText(err)), closeButton());
  }
}

function closeButton(): HTMLElement {
  const button = h("button", { class: "close", "aria-label": "Close" }, "×");
  button.addEventListener("click", () => dialog.close());
  return button;
}

dialog.addEventListener("click", (e) => {
  if (e.target === dialog) dialog.close(); // backdrop
});

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Renders `load()` into `section`, replacing what was there. */
async function showRecommendations(
  section: HTMLElement,
  load: () => Promise<RecommendationsResponse>,
  extra?: (r: RecommendationsResponse) => Node | null,
): Promise<void> {
  section.replaceChildren(h("p", { class: "status" }, "Loading recommendations…"));
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
  main.append(h("h1", {}, "Recommendations for ", h("code", {}, userId)), section);
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
    placeholder: "Loading the game catalog…",
    "aria-label": "Search games",
    autocomplete: "off",
    disabled: "",
  });
  const results = h("ul", { class: "results", role: "listbox" });
  const chips = h("ul", { class: "chips" });
  const recommend = h("button", { class: "primary" }, "Recommend");
  const clear = h("button", {}, "Clear");
  const section = h("section", {});

  main.append(
    h("h1", {}, "Find your next game"),
    h(
      "p",
      { class: "lead" },
      "Search the games you liked, then get recommendations from the two-tower model. ",
      "Or open a player's recommendations with the Steam id box above.",
    ),
    h("div", { class: "search" }, input, results),
    h("div", { class: "liked" }, chips, h("div", { class: "actions" }, recommend, clear)),
    section,
  );

  const renderLiked = () => {
    chips.replaceChildren(
      ...liked.games.map((g) => {
        const remove = h("button", { "aria-label": `Remove ${g.name}` }, "×");
        remove.addEventListener("click", () => {
          liked.remove(g.game_id);
          renderLiked();
        });
        return h("li", {}, g.name, remove);
      }),
    );
    if (!liked.games.length) chips.append(h("li", { class: "hint" }, "No games picked yet."));
    recommend.toggleAttribute("disabled", !liked.games.length);
    clear.toggleAttribute("disabled", !liked.games.length);
  };

  const pick = (hit: GameHit) => {
    if (liked.games.length >= MAX_LIKED_GAMES) return;
    liked.add(hit);
    renderLiked();
    input.value = "";
    results.replaceChildren();
    input.focus();
  };

  let latest = 0;
  input.addEventListener("input", async () => {
    const seq = ++latest;
    const hits = await search(input.value);
    if (seq !== latest) return; // a newer query answered first
    results.replaceChildren(
      ...hits.map((hit) => {
        const item = h(
          "li",
          { role: "option", tabindex: "0" },
          h("span", {}, hit.name),
          h("small", {}, `${hit.reviews.toLocaleString()} reviews`),
        );
        item.addEventListener("click", () => pick(hit));
        item.addEventListener("keydown", (e) => {
          if (e.key === "Enter") pick(hit);
        });
        return item;
      }),
    );
  });

  recommend.addEventListener("click", () =>
    showRecommendations(
      section,
      () => onlineRecommendations(liked.games.map((g) => g.game_id)),
      (r) =>
        r.ignored_game_ids?.length
          ? h(
              "p",
              { class: "hint" },
              `${r.ignored_game_ids.length} picked game(s) are unknown to the model.`,
            )
          : null,
    ),
  );
  clear.addEventListener("click", () => {
    liked.clear();
    renderLiked();
    void showRecommendations(section, popular);
  });

  renderLiked();
  startSearch().then(
    (size) => {
      input.removeAttribute("disabled");
      input.placeholder = `Search ${size.toLocaleString()} games…`;
    },
    (err) => {
      input.placeholder = "Search is unavailable";
      results.replaceChildren(h("li", { class: "error" }, errorText(err)));
    },
  );
  void showRecommendations(section, popular);
}

render();
