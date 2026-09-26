// @vitest-environment node
import { createHash } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  headerImage,
  onlineRecommendations,
  searchIndex,
  sha256Hex,
  userRecommendations,
} from "../src/api";
import { index, personalized } from "./fixtures";

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn(
    async (_url: string, _init?: RequestInit) => new Response(JSON.stringify(body), { status }),
  );
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

describe("api", () => {
  it("hashes the body like SigV4 expects (hex SHA-256)", async () => {
    const body = '{"liked_game_ids":[620]}';
    expect(await sha256Hex(body)).toBe(createHash("sha256").update(body).digest("hex"));
  });

  it("POSTs liked games with the hash of the exact body sent", async () => {
    const fetch = mockFetch(200, { ...personalized, source: "online" });
    const response = await onlineRecommendations([620, 400]);
    expect(response.source).toBe("online");
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe("/api/recommendations");
    expect(init?.method).toBe("POST");
    const body = init?.body as string;
    expect(JSON.parse(body)).toEqual({ liked_game_ids: [620, 400], limit: 30, details: true });
    const headers = init?.headers as Record<string, string>;
    expect(headers["x-amz-content-sha256"]).toBe(createHash("sha256").update(body).digest("hex"));
    expect(Object.keys(headers).map((h) => h.toLowerCase())).not.toContain("authorization");
  });

  it("asks for a user's list with details, id kept as a string", async () => {
    const fetch = mockFetch(200, personalized);
    await userRecommendations("76561198312196006");
    expect(fetch.mock.calls[0][0]).toBe(
      "/api/users/76561198312196006/recommendations?limit=30&details=true",
    );
  });

  it("turns API errors into a Spanish message, keeping the server's in detail", async () => {
    mockFetch(503, { error: "online model not available", detail: "no bundle published yet" });
    const err = await onlineRecommendations([1]).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(503);
    expect(err.message).toBe("El modelo aún no está disponible. Inténtalo más tarde.");
    expect(err.detail).toBe("online model not available: no bundle published yet");
    mockFetch(500, "not json");
    expect((await onlineRecommendations([1]).catch((e) => e)).message).toMatch(/HTTP 500/);
  });

  it("reports network failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))),
    );
    const err = await userRecommendations("1").catch((e) => e);
    expect([err.status, err.message]).toEqual([0, "No se pudo conectar con el servidor."]);
  });

  it("rejects a search index of another format", async () => {
    mockFetch(200, index);
    expect((await searchIndex()).games).toHaveLength(6);
    mockFetch(200, { ...index, format_version: 2 });
    await expect(searchIndex()).rejects.toMatchObject({ detail: "format 2" });
  });

  it("falls back to Steam's CDN image without details", () => {
    expect(headerImage(personalized.recommendations[0])).toBe("https://cdn.example/620/header.jpg");
    expect(headerImage({ game_id: 400 })).toMatch(/\/apps\/400\/header\.jpg$/);
  });
});
