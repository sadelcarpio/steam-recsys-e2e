import type { RecommendationsResponse, SearchIndex } from "../src/api";

// Shaped like the serving API's JSON (exclude_none: absent optional fields).
export const personalized: RecommendationsResponse = {
  source: "personalized",
  user_id: "76561198312196006",
  model_id: "local-e61c060-val1",
  generated_at: "2026-09-25T20:00:00Z",
  reranked: true,
  rerank_model: "us.amazon.nova-2-lite-v1:0",
  recommendations: [
    {
      rank: 1,
      game_id: 620,
      name: "Portal 2",
      score: 0.81,
      explanation: "You loved <b>puzzle</b> games like Portal.",
      details: {
        game_id: 620,
        name: "Portal 2",
        short_description: "The sequel to Portal.",
        header_image: "https://cdn.example/620/header.jpg",
        release_date: "Apr 18, 2011",
        is_free: false,
        price: 9.99,
        developers: ["Valve"],
        publishers: ["Valve"],
        genres: ["Action", "Adventure", "Puzzle", "Casual"],
        categories: [],
      },
    },
    { rank: 2, game_id: 400, name: "Portal", score: 0.7 },
  ],
};

export const index: SearchIndex = {
  format_version: 1,
  model_id: "m",
  generated_at: "2026-09-25T20:00:00Z",
  games: [
    [730, "Counter-Strike 2", 90000],
    [620, "Portal 2", 50000],
    [400, "Portal", 20000],
    [12345, "Portal Knights", 3000],
    [99999, "Portal Pals Obscure", 1],
    [440, "Team Fortress 2", 40000],
  ],
};
