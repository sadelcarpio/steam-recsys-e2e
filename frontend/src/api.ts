// Client of the serving API (serving/src/steam_serving/contracts.py), reached through the same
// CloudFront distribution under /api, and of the search index (/data/games.json, written by
// inference: inference/src/steam_inference/contracts.py `SearchIndex`).

export interface GameDetails {
  game_id: number;
  name: string;
  short_description?: string;
  header_image?: string;
  release_date?: string;
  is_free?: boolean;
  price?: number;
  developers?: string[];
  publishers?: string[];
  genres?: string[];
  categories?: string[];
}

export interface Recommendation {
  rank: number;
  game_id: number;
  name: string;
  score: number;
  explanation?: string;
  details?: GameDetails;
}

export interface RecommendationsResponse {
  source: "personalized" | "popular" | "online";
  user_id?: string | null;
  model_id: string;
  generated_at: string;
  reranked: boolean;
  rerank_model?: string;
  recommendations: Recommendation[];
  used_game_ids?: number[];
  ignored_game_ids?: number[];
}

export interface SearchIndex {
  format_version: number;
  model_id: string;
  generated_at: string;
  games: [appid: number, name: string, reviews: number][];
}

export const SEARCH_INDEX_FORMAT = 1;
export const LIMIT = 30; // serving MAX_LIMIT (inference writes 30 per user)
export const MAX_LIKED_GAMES = 100; // serving MAX_LIKED_GAMES
export const USER_ID_PATTERN = /^[0-9]{1,20}$/; // Steam 64-bit account id (a string)

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { error?: string; detail?: string };
      if (body.error) message = body.detail ? `${body.error}: ${body.detail}` : body.error;
    } catch {
      // not JSON (e.g. a CloudFront error page)
    }
    throw new ApiError(response.status, message);
  }
  return (await response.json()) as T;
}

export async function userRecommendations(userId: string): Promise<RecommendationsResponse> {
  const id = encodeURIComponent(userId);
  return json(await fetch(`/api/users/${id}/recommendations?limit=${LIMIT}&details=true`));
}

export async function popular(): Promise<RecommendationsResponse> {
  return json(await fetch(`/api/popular?limit=${LIMIT}&details=true`));
}

export async function game(gameId: number): Promise<GameDetails> {
  return json(await fetch(`/api/games/${gameId}`));
}

// CloudFront signs requests to the Function URL (OAC) but does not hash the body: a POST must
// carry the SHA-256 of its exact body, hex encoded.
export async function sha256Hex(body: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(body));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function onlineRecommendations(
  likedGameIds: number[],
): Promise<RecommendationsResponse> {
  const body = JSON.stringify({ liked_game_ids: likedGameIds, limit: LIMIT, details: true });
  return json(
    await fetch("/api/recommendations", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-amz-content-sha256": await sha256Hex(body),
      },
      body,
    }),
  );
}

export async function searchIndex(): Promise<SearchIndex> {
  const index = await json<SearchIndex>(await fetch("/data/games.json"));
  if (index.format_version !== SEARCH_INDEX_FORMAT) {
    throw new Error(`search index format ${index.format_version}, expected ${SEARCH_INDEX_FORMAT}`);
  }
  return index;
}

// Steam's CDN image of a game without stored details.
export function headerImage(game: { game_id: number; details?: GameDetails }): string {
  return (
    game.details?.header_image ??
    `https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/${game.game_id}/header.jpg`
  );
}
