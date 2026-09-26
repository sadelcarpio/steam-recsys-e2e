// The games picked on the discover page, most recent first (the order the model reads the
// history in). Remembered in localStorage when available: a convenience, never required.
const KEY = "recsys.liked";

export interface LikedGame {
  game_id: number;
  name: string;
}

export class LikedGames {
  games: LikedGame[];

  constructor(private readonly storage: Storage | null = safeStorage()) {
    this.games = this.load();
  }

  add(game: LikedGame): void {
    this.games = [
      { game_id: game.game_id, name: game.name },
      ...this.games.filter((g) => g.game_id !== game.game_id),
    ];
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
