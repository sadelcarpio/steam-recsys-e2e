// The CloudFront Functions (infrastructure/cloudfront/*.js, cloudfront-js-2.0 runtime): plain
// ES5-style `handler(event)` scripts, evaluated here as they are uploaded.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

type Handler = (event: { request: { uri: string } }) => { uri: string };

function load(name: string): Handler {
  const code = readFileSync(resolve(__dirname, "../../infrastructure/cloudfront", name), "utf8");
  return new Function(`${code}; return handler;`)() as Handler;
}

const uri = (handler: Handler, path: string) => handler({ request: { uri: path } }).uri;

describe("cloudfront functions", () => {
  it("spa: app routes serve index.html, files pass through", () => {
    const spa = load("spa.js");
    expect(uri(spa, "/")).toBe("/index.html");
    expect(uri(spa, "/u/76561198312196006")).toBe("/index.html");
    expect(uri(spa, "/assets/index-abc.js")).toBe("/assets/index-abc.js");
    expect(uri(spa, "/favicon.ico")).toBe("/favicon.ico");
  });

  it("strip_prefix: the origin sees its own paths", () => {
    const strip = load("strip_prefix.js");
    expect(uri(strip, "/api/users/1/recommendations")).toBe("/users/1/recommendations");
    expect(uri(strip, "/api/popular")).toBe("/popular");
    expect(uri(strip, "/api/recommendations")).toBe("/recommendations");
    expect(uri(strip, "/data/games.json")).toBe("/games.json");
    expect(uri(strip, "/api")).toBe("/");
  });
});
