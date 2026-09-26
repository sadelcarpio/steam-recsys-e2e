// Loads the search index and answers queries off the main thread.
// in:  { id, query }       out: { type: "ready", size } | { type: "error", message }
//                               | { type: "results", id, hits }
import { searchIndex } from "./api";
import { buildSearch, type GameSearch } from "./search";

const ready: Promise<GameSearch> = searchIndex().then(buildSearch);

ready.then(
  (s) => postMessage({ type: "ready", size: s.size }),
  (err: unknown) => postMessage({ type: "error", message: String(err) }),
);

onmessage = async (event: MessageEvent<{ id: number; query: string }>) => {
  const search = await ready;
  postMessage({ type: "results", id: event.data.id, hits: search.search(event.data.query) });
};
