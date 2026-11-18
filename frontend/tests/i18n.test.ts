import { afterEach, describe, expect, it, vi } from "vitest";
import { MESSAGES, resolveLocale } from "../src/i18n";
import { personalized } from "./fixtures";

describe("locale", () => {
  it("takes ?lang= first, then the remembered one, else English", () => {
    expect(resolveLocale("?lang=es", "en")).toBe("es");
    expect(resolveLocale("?lang=en", "es")).toBe("en");
    expect(resolveLocale("", "es")).toBe("es");
    expect(resolveLocale("?lang=fr", null)).toBe("en");
    expect(resolveLocale("", "xx")).toBe("en");
    expect(resolveLocale("", null)).toBe("en");
  });

  it("has every text in both languages", () => {
    const keys = (o: object): string[] =>
      Object.entries(o)
        .flatMap(([k, v]) => (typeof v === "object" ? keys(v).map((sub) => `${k}.${sub}`) : [k]))
        .sort();
    expect(keys(MESSAGES.en)).toEqual(keys(MESSAGES.es));
  });
});

describe("English", () => {
  afterEach(() => {
    localStorage.setItem("lang", "es"); // back to the other suites' locale
    vi.resetModules();
  });

  it("renders the summaries, dates and genres in English", async () => {
    localStorage.setItem("lang", "en");
    vi.resetModules();
    const { responseSummary } = await import("../src/cards");
    const { genreName, t } = await import("../src/i18n");
    expect(responseSummary({ ...personalized, reranked: false })).toBe(
      "Personalized for this player (two-tower model: local-e61c060-val1), " +
        "explained by an LLM (September 25, 2026).",
    );
    expect(responseSummary({ ...personalized, source: "online", ranker_id: "k1" })).toBe(
      "Computed now from the games you picked (retrieval: local-e61c060-val1, ranking: k1).",
    );
    expect(genreName("Strategy")).toBe("Strategy");
    expect(t.userTitle(1234)).toBe("User #1,234");
  });
});
