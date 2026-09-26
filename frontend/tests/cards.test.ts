import { describe, expect, it, vi } from "vitest";
import { cardGrid, detailsView, priceLabel, responseSummary } from "../src/cards";
import { personalized } from "./fixtures";

describe("cards", () => {
  it("renders one card per recommendation with image, rank, tags and explanation", () => {
    const grid = cardGrid(personalized, () => {});
    const cards = grid.querySelectorAll(".card");
    expect(cards).toHaveLength(2);
    const first = cards[0];
    expect(first.querySelector("img")?.getAttribute("src")).toBe(
      "https://cdn.example/620/header.jpg",
    );
    expect(first.querySelector("h3")?.textContent).toBe("#1 Portal 2");
    expect([...first.querySelectorAll(".tags li")].map((li) => li.textContent)).toEqual([
      "Acción",
      "Aventura",
      "Puzzle", // not a Steam genre: shown as is
      "$9.99",
    ]);
    expect(cards[1].querySelector(".explanation")).toBeNull();
  });

  it("never renders API text as HTML", () => {
    const grid = cardGrid(personalized, () => {});
    const explanation = grid.querySelector(".explanation")!;
    expect(explanation.textContent).toBe("You loved <b>puzzle</b> games like Portal.");
    expect(explanation.querySelector("b")).toBeNull();
  });

  it("opens a card on click and Enter", () => {
    const open = vi.fn();
    const card = cardGrid(personalized, open).querySelector<HTMLElement>(".card")!;
    card.click();
    card.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    expect(open).toHaveBeenCalledTimes(2);
    expect(open.mock.calls[0][0].game_id).toBe(620);
  });

  it("shows an empty state", () => {
    const grid = cardGrid({ ...personalized, recommendations: [] }, () => {});
    expect(grid.textContent).toBe("No hay recomendaciones.");
  });

  it("describes where the list comes from", () => {
    expect(responseSummary(personalized)).toMatch(/reordenadas y explicadas por un LLM/);
    expect(responseSummary(personalized)).toMatch(/septiembre de 2026/);
    expect(responseSummary({ ...personalized, source: "popular" })).toMatch(
      /^Aún no hay recomendaciones para este jugador/,
    );
    expect(responseSummary({ ...personalized, source: "popular", user_id: null })).toMatch(
      /^Juegos más populares/,
    );
    expect(responseSummary({ ...personalized, source: "online" })).toMatch(/juegos que elegiste/);
  });

  it("renders details with facts and a Steam link", () => {
    const view = detailsView(personalized.recommendations[0].details!, "why");
    expect(view.querySelector("h2")?.textContent).toBe("Portal 2");
    expect([...view.querySelectorAll("dt")].map((d) => d.textContent)).toEqual([
      "Lanzamiento",
      "Precio",
      "Desarrollador",
      "Editor",
      "Géneros",
    ]);
    expect(view.querySelector("a")?.getAttribute("href")).toBe(
      "https://store.steampowered.com/app/620/",
    );
  });

  it("labels prices", () => {
    expect(priceLabel(undefined)).toBeNull();
    expect(priceLabel({ game_id: 1, name: "x", is_free: true })).toBe("Gratis");
    expect(priceLabel({ game_id: 1, name: "x" })).toBeNull();
  });
});
