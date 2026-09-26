// Game search over the search index, in the browser (MiniSearch): name prefix + fuzzy match,
// ranked by relevance boosted by the game's review count. Built in a Web Worker
// (search.worker.ts) so indexing ~100k names does not block the page.
import MiniSearch from "minisearch";
import type { SearchIndex } from "./api";

export interface GameHit {
  game_id: number;
  name: string;
  reviews: number;
}

export interface GameSearch {
  search(query: string, limit?: number): GameHit[];
  readonly size: number;
}

export function buildSearch(index: SearchIndex): GameSearch {
  const engine = new MiniSearch<GameHit>({
    idField: "game_id",
    fields: ["name"],
    storeFields: ["game_id", "name", "reviews"],
    searchOptions: {
      prefix: true,
      fuzzy: (term) => (term.length > 3 ? 0.2 : false),
      combineWith: "AND",
      boostDocument: (_id, _term, stored) => 1 + Math.log10(1 + Number(stored?.reviews ?? 0)),
    },
  });
  engine.addAll(index.games.map(([game_id, name, reviews]) => ({ game_id, name, reviews })));
  return {
    size: engine.documentCount,
    search(query, limit = 10) {
      const q = query.trim();
      if (!q) return [];
      return engine
        .search(q)
        .slice(0, limit)
        .map((r) => ({
          game_id: r.game_id as number,
          name: r.name as string,
          reviews: r.reviews as number,
        }));
    },
  };
}
