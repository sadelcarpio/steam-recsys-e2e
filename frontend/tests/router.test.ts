import { describe, expect, it } from "vitest";
import { neighbours, parseRoute, randomUser, RANDOM_USER_RANGE, userPath } from "../src/router";

describe("router", () => {
  it.each([
    ["/", { page: "discover" }],
    ["/index.html", { page: "discover" }],
    ["/u/12", { page: "user", userIdx: 12 }],
    ["/u/12/", { page: "user", userIdx: 12 }],
    ["/u/0", { page: "not-found" }], // user numbers start at 1
    ["/u/012", { page: "not-found" }],
    ["/u/76561198312196006", { page: "not-found" }], // Steam ids are not user numbers
    ["/u/__popular__", { page: "not-found" }],
    ["/nope", { page: "not-found" }],
  ])("%s", (path, route) => expect(parseRoute(path)).toEqual(route));

  it("round-trips user paths", () => {
    expect(parseRoute(userPath(42))).toEqual({ page: "user", userIdx: 42 });
  });
});

describe("user navigation", () => {
  it("links the neighbours inside 1..max", () => {
    expect(neighbours(1, 5)).toEqual({ next: 2 });
    expect(neighbours(3, 5)).toEqual({ prev: 2, next: 4 });
    expect(neighbours(5, 5)).toEqual({ prev: 4 });
    expect(neighbours(9, 5)).toEqual({ prev: 5 }); // past the end (an older link)
    expect(neighbours(1, 1)).toEqual({});
  });

  it("picks a random user among the reranked ones, never the current one", () => {
    expect(randomUser(1_000_000, undefined, () => 0.9999)).toBe(RANDOM_USER_RANGE);
    expect(randomUser(10, undefined, () => 0)).toBe(1);
    expect(randomUser(10, 1, () => 0)).toBe(2);
    expect(randomUser(10, 10, () => 0.9999)).toBe(1);
    expect(randomUser(1, 1)).toBe(1);
  });
});
