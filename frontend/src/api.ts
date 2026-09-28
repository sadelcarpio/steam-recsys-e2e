// Client of the serving API (serving/src/steam_serving/contracts.py), reached through the same
// CloudFront distribution under /api, and of the search index (/data/games.json, written by
// inference: inference/src/steam_inference/contracts.py `SearchIndex`). Errors carry a Spanish
// message for the page; the server's own (English) error stays in `detail`, for the console.
import { t } from "./i18n";

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
// The user tower reads the last 5 liked games (training USER_HISTORY_LENGTH): more would be
// ignored. Serving accepts up to 100.
export const MAX_LIKED_GAMES = 5;
export const USER_ID_PATTERN = /^[0-9]{1,20}$/; // Steam 64-bit account id (a string)

export class ApiError extends Error {
  constructor(
    readonly status: number, // 0: no response (network)
    message: string,
    readonly detail?: string,
  ) {
    super(message);
  }
}

function statusMessage(status: number): string {
  if (status === 400 || status === 404 || status === 503) return t.errors[status];
  return t.errors.other(status);
}

async function request(url: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (err) {
    throw new ApiError(0, t.errors.network, String(err));
  }
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail: string | undefined;
    try {
      const body = (await response.json()) as { error?: string; detail?: string };
      if (body.error) detail = body.detail ? `${body.error}: ${body.detail}` : body.error;
    } catch {
      // not JSON (e.g. a CloudFront error page)
    }
    throw new ApiError(response.status, statusMessage(response.status), detail);
  }
  return (await response.json()) as T;
}

export async function userRecommendations(userId: string): Promise<RecommendationsResponse> {
  const id = encodeURIComponent(userId);
  return json(await request(`/api/users/${id}/recommendations?limit=${LIMIT}&details=true`));
}

export async function popular(): Promise<RecommendationsResponse> {
  return json(await request(`/api/popular?limit=${LIMIT}&details=true`));
}

export async function game(gameId: number): Promise<GameDetails> {
  return json(await request(`/api/games/${gameId}`));
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
    await request("/api/recommendations", {
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
  const index = await json<SearchIndex>(await request("/data/games.json"));
  if (index.format_version !== SEARCH_INDEX_FORMAT) {
    throw new ApiError(200, t.errors.searchIndex, `format ${index.format_version}`);
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
