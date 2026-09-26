import { describe, expect, it } from "vitest";
import { LikedGames } from "../src/liked";

describe("liked games", () => {
  it("keeps the most recent pick first, without duplicates, across page loads", () => {
    const liked = new LikedGames(localStorage);
    liked.clear();
    liked.add({ game_id: 1, name: "A" });
    liked.add({ game_id: 2, name: "B" });
    liked.add({ game_id: 1, name: "A" });
    expect(liked.games.map((g) => g.game_id)).toEqual([1, 2]);
    expect(new LikedGames(localStorage).games.map((g) => g.game_id)).toEqual([1, 2]);
    liked.remove(1);
    expect(new LikedGames(localStorage).games).toEqual([{ game_id: 2, name: "B" }]);
  });

  it("ignores corrupt storage and works without storage", () => {
    localStorage.setItem("recsys.liked", "{not json");
    expect(new LikedGames(localStorage).games).toEqual([]);
    localStorage.setItem(
      "recsys.liked",
      JSON.stringify([{ game_id: "x" }, { game_id: 3, name: "C" }]),
    );
    expect(new LikedGames(localStorage).games).toEqual([{ game_id: 3, name: "C" }]);
    const memory = new LikedGames(null);
    memory.add({ game_id: 4, name: "D" });
    expect(memory.games).toHaveLength(1);
  });
});
