// The games picked on the discover page, most recent first (the order the model reads the
// history in), at most `max`. Remembered in localStorage when available: a convenience, never
// required.
import { MAX_LIKED_GAMES } from "./api";

const KEY = "recsys.liked";

export interface LikedGame {
  game_id: number;
  name: string;
}

export class LikedGames {
  games: LikedGame[];

  constructor(
    private readonly storage: Storage | null = safeStorage(),
    readonly max: number = MAX_LIKED_GAMES,
  ) {
    this.games = this.load().slice(0, max);
  }

  get full(): boolean {
    return this.games.length >= this.max;
  }

  /** false (and nothing changes) when the list is full and `game` is not in it yet. */
  add(game: LikedGame): boolean {
    const present = this.games.some((g) => g.game_id === game.game_id);
    if (this.full && !present) return false;
    this.games = [
      { game_id: game.game_id, name: game.name },
      ...this.games.filter((g) => g.game_id !== game.game_id),
    ];
    this.save();
    return true;
  }

  /** Moves the game at position `from` to position `to` (0 = most recent); the others shift. */
  move(from: number, to: number): void {
    const last = this.games.length - 1;
    if (from < 0 || from > last) return;
    to = Math.min(Math.max(to, 0), last);
    if (from === to) return;
    const games = [...this.games];
    const [game] = games.splice(from, 1);
    games.splice(to, 0, game);
    this.games = games;
    this.save();
  }

  remove(gameId: number): void {
    this.games = this.games.filter((g) => g.game_id !== gameId);
    this.save();
  }

  clear(): void {
    this.games = [];
    this.save();
  }

  private load(): LikedGame[] {
    try {
      const parsed: unknown = JSON.parse(this.storage?.getItem(KEY) ?? "[]");
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(
        (g): g is LikedGame => typeof g?.game_id === "number" && typeof g?.name === "string",
      );
    } catch {
      return [];
    }
  }

  private save(): void {
    try {
      this.storage?.setItem(KEY, JSON.stringify(this.games));
    } catch {
      // storage full or blocked: the list still works for this page
    }
  }
}

function safeStorage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}
