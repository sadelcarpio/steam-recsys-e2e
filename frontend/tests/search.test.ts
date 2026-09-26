import { describe, expect, it } from "vitest";
import { buildSearch } from "../src/search";
import { index } from "./fixtures";

describe("search", () => {
  const search = buildSearch(index);

  it("indexes every game", () => expect(search.size).toBe(6));

  it("matches name prefixes, ranked by relevance only", () => {
    const names = search.search("port").map((h) => h.name);
    expect(names[0]).toBe("Portal"); // the closest match, not the most reviewed ("Portal 2")
    expect(names).toEqual(expect.arrayContaining(["Portal 2", "Portal Knights"]));
    // review counts don't reorder equally relevant names
    const pals = search.search("portal pals").map((h) => h.name);
    expect(pals[0]).toBe("Portal Pals Obscure");
  });

  it("requires every word and tolerates typos in longer words", () => {
    expect(search.search("team fort").map((h) => h.game_id)).toEqual([440]);
    expect(search.search("counter strke").map((h) => h.game_id)).toEqual([730]);
  });

  it("returns hits with their review counts, limited", () => {
    expect(search.search("portal", 2)).toHaveLength(2);
    expect(search.search("counter")[0]).toEqual({
      game_id: 730,
      name: "Counter-Strike 2",
      reviews: 90000,
    });
    expect(search.search("   ")).toEqual([]);
  });
});
