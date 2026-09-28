// Two pages: "/" (discover: search + liked games) and "/u/<steam id>" (a user's recommendations).
import { USER_ID_PATTERN } from "./api";

export type Route = { page: "discover" } | { page: "user"; userId: string } | { page: "not-found" };

export function parseRoute(pathname: string): Route {
  const path = pathname.replace(/\/+$/, "") || "/";
  if (path === "/" || path === "/index.html") return { page: "discover" };
  const match = /^\/u\/([^/]+)$/.exec(path);
  if (match) {
    const userId = decodeURIComponent(match[1]);
    if (USER_ID_PATTERN.test(userId)) return { page: "user", userId };
  }
  return { page: "not-found" };
}

export function userPath(userId: string): string {
  return `/u/${encodeURIComponent(userId)}`;
}
