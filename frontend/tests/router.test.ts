import { describe, expect, it } from "vitest";
import { parseRoute, userPath } from "../src/router";

describe("router", () => {
  it.each([
    ["/", { page: "discover" }],
    ["/index.html", { page: "discover" }],
    ["/u/76561198312196006", { page: "user", userId: "76561198312196006" }],
    ["/u/76561198312196006/", { page: "user", userId: "76561198312196006" }],
    ["/u/__popular__", { page: "not-found" }],
    ["/u/123456789012345678901", { page: "not-found" }],
    ["/nope", { page: "not-found" }],
  ])("%s", (path, route) => expect(parseRoute(path)).toEqual(route));

  it("round-trips user paths", () => {
    expect(parseRoute(userPath("42"))).toEqual({ page: "user", userId: "42" });
  });
});
