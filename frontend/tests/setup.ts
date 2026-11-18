// Runs before every test file, before the app modules load: i18n.ts picks the locale at import,
// so the suites that assert the Spanish texts get them. tests/i18n.test.ts switches to English.
// Node-environment suites (api.test.ts) have no localStorage: a minimal one stands in.
if (typeof localStorage === "undefined") {
  const items = new Map<string, string>();
  globalThis.localStorage = {
    getItem: (key: string) => items.get(key) ?? null,
    setItem: (key: string, value: string) => void items.set(key, value),
    removeItem: (key: string) => void items.delete(key),
    clear: () => items.clear(),
    key: (i: number) => [...items.keys()][i] ?? null,
    get length() {
      return items.size;
    },
  };
}
localStorage.setItem("lang", "es");
