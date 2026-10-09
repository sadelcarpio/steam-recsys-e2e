// Two pages: "/" (discover: search + liked games) and "/u/<user number>" (a demo user's
// recommendations, spec 13: 1 = the most active user; never a Steam id).
import { USER_IDX_PATTERN } from "./api";

export type Route =
  { page: "discover" } | { page: "user"; userIdx: number } | { page: "not-found" };

export function parseRoute(pathname: string): Route {
  const path = pathname.replace(/\/+$/, "") || "/";
  if (path === "/" || path === "/index.html") return { page: "discover" };
  const match = /^\/u\/([^/]+)$/.exec(path);
  if (match) {
    const userIdx = decodeURIComponent(match[1]);
    if (USER_IDX_PATTERN.test(userIdx)) return { page: "user", userIdx: Number(userIdx) };
  }
  return { page: "not-found" };
}

export function userPath(userIdx: number): string {
  return `/u/${userIdx}`;
}

// The users with LLM explanations are the most active ones (inference RERANK_MAX_USERS).
export const RANDOM_USER_RANGE = 1000;

/** Neighbours of user `userIdx` among 1..maxUser (absent at the ends). */
export function neighbours(userIdx: number, maxUser: number): { prev?: number; next?: number } {
  return {
    ...(userIdx > 1 ? { prev: Math.min(userIdx - 1, maxUser) } : {}),
    ...(userIdx < maxUser ? { next: userIdx + 1 } : {}),
  };
}

/** A random user among the first RANDOM_USER_RANGE (the reranked ones), other than `current`. */
export function randomUser(maxUser: number, current?: number, random = Math.random): number {
  const range = Math.min(maxUser, RANDOM_USER_RANGE);
  if (range <= 1) return 1;
  let pick = 1 + Math.floor(random() * range);
  if (pick === current) pick = (pick % range) + 1;
  return pick;
}
