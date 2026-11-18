// Every text the app renders itself, in English (the default) and Spanish. Data from the API
// (game names, descriptions, LLM explanations) is shown as is; Steam's genre names are
// translated when known (Spanish only: the data has them in English).
//
// The locale comes from `?lang=en|es` (remembered in localStorage), else English. Switching
// reloads the page: `t` is chosen once, when this module loads.

export type Locale = "en" | "es";
export const LOCALES: Locale[] = ["en", "es"];
export const LOCALE_KEY = "lang";

export function resolveLocale(search: string, stored: string | null): Locale {
  const asked = new URLSearchParams(search).get(LOCALE_KEY);
  for (const candidate of [asked, stored]) {
    if (candidate === "en" || candidate === "es") return candidate;
  }
  return "en";
}

function storedLocale(): string | null {
  try {
    return localStorage.getItem(LOCALE_KEY);
  } catch {
    return null; // storage blocked (private mode, sandboxed previews)
  }
}

export function rememberLocale(locale: Locale): void {
  try {
    localStorage.setItem(LOCALE_KEY, locale);
  } catch {
    // not remembered: the ?lang= link still applies to this page
  }
}

export const LOCALE: Locale = resolveLocale(
  typeof location === "undefined" ? "" : location.search,
  storedLocale(),
);

const es = {
  brand: "🎮 Recomendador de juegos de Steam",
  userIdPlaceholder: "Usuario n.º (1 = el más activo)",
  userIdLabel: "Número de usuario",
  userIdSubmit: "Ver recomendaciones",
  userIdInvalid: "Un número de usuario: 1, 2, 3…",
  notFound: "Página no encontrada. ",
  goHome: "Volver al inicio",
  loading: "Cargando…",
  loadingRecommendations: "Cargando recomendaciones…",
  close: "Cerrar",
  userTitle: (n: number) => `Usuario n.º ${n.toLocaleString("es")}`,
  userOf: (max: number) => ` de ${max.toLocaleString("es")}`,
  userHint:
    "Los usuarios se numeran por actividad: el 1 es quien más reseñas escribió. La numeración " +
    "se recalcula en cada actualización de los datos, y los IDs de Steam no se guardan.",
  previousUser: "← Anterior",
  nextUser: "Siguiente →",
  discoverTitle: "Encuentra tu próximo juego",
  discoverLead: (max: number) =>
    `Busca y elige hasta ${max} juegos que te gustaron, y el modelo de dos torres te ` +
    "recomendará otros. También puedes ver las recomendaciones de un jugador por su número, " +
    "arriba (1 = el más activo).",
  searchLoading: "Cargando el catálogo de juegos…",
  searchPlaceholder: (size: number) => `Buscar entre ${size.toLocaleString("es")} juegos…`,
  searchFull: (max: number) => `Máximo ${max} juegos: quita uno para elegir otro`,
  searchLabel: "Buscar juegos",
  searchUnavailable: "La búsqueda no está disponible",
  noResults: "Sin resultados",
  picked: (n: number, max: number) => `${n} de ${max} juegos elegidos`,
  // shown once there are 2+ picks: the chips' numbers are the model's history order
  pickOrder: " · 1 = el más reciente. Arrástralos (o usa las flechas) para cambiar el orden.",
  pickLabel: (position: number, name: string) => `${position}.º ${name}`,
  noneChosen: "Aún no elegiste ningún juego.",
  remove: (name: string) => `Quitar ${name}`,
  recommend: "Recomendar",
  clear: "Limpiar",
  startHint: "Elige al menos un juego para ver recomendaciones.",
  unknownPicked: (n: number) =>
    n === 1
      ? "1 juego elegido no está en el catálogo del modelo."
      : `${n} juegos elegidos no están en el catálogo del modelo.`,
  noRecommendations: "No hay recomendaciones.",
  likedTitle: "Sus últimos juegos favoritos",
  likedHint: "Las reseñas positivas más recientes de este jugador: el modelo parte de ellas.",
  recommendationsTitle: "Recomendaciones",
  explainedBy: (model: string) => `modelo: ${model}`,
  // `models`: modelsLabel's text
  summaryReranked: (when: string, models: string) =>
    `Personalizadas para este jugador (${models}), reordenadas y explicadas por un LLM (${when}).`,
  summaryPersonalized: (when: string, models: string) =>
    `Personalizadas para este jugador (${models}, ${when}).`,
  summaryExplained: (when: string, models: string) =>
    `Personalizadas para este jugador (${models}), explicadas por un LLM (${when}).`,
  summaryOnline: (models: string) =>
    `Calculadas ahora a partir de los juegos que elegiste (${models}).`,
  // the retriever (two-tower model) and, when one ordered the list, the ranker
  modelsLabel: (retriever: string, ranker?: string) =>
    ranker ? `retrieval: ${retriever}, ranking: ${ranker}` : `modelo de dos torres: ${retriever}`,
  summaryFallback: (when: string) =>
    `Aún no hay recomendaciones para este jugador: se muestran los juegos más populares (${when}).`,
  summaryPopular: (when: string) => `Juegos más populares en las reseñas recientes (${when}).`,
  free: "Gratis",
  released: "Lanzamiento",
  price: "Precio",
  developer: "Desarrollador",
  publisher: "Editor",
  genres: "Géneros",
  viewOnSteam: "Ver en Steam ↗",
  errors: {
    400: "La solicitud no es válida.",
    404: "No se encontró lo que buscabas.",
    503: "El modelo aún no está disponible. Inténtalo más tarde.",
    other: (status: number) => `Error del servidor (HTTP ${status}). Inténtalo de nuevo.`,
    network: "No se pudo conectar con el servidor.",
    unknown: "Algo salió mal. Inténtalo de nuevo.",
    searchIndex: "El índice de búsqueda tiene un formato inesperado.",
  },
  // the header link to the other language, and the page's meta description
  languageSwitch: "English",
  description:
    "Recomendaciones de juegos de Steam con un modelo de dos torres, un ranker y explicaciones de un LLM.",
};

type Messages = typeof es;

const en: Messages = {
  brand: "🎮 Steam Game Recommender",
  userIdPlaceholder: "User # (1 = most active)",
  userIdLabel: "User number",
  userIdSubmit: "See recommendations",
  userIdInvalid: "A user number: 1, 2, 3…",
  notFound: "Page not found. ",
  goHome: "Back to the start",
  loading: "Loading…",
  loadingRecommendations: "Loading recommendations…",
  close: "Close",
  userTitle: (n: number) => `User #${n.toLocaleString("en")}`,
  userOf: (max: number) => ` of ${max.toLocaleString("en")}`,
  userHint:
    "Users are numbered by activity: #1 wrote the most reviews. The numbers are recomputed " +
    "on every data update, and Steam IDs are not stored.",
  previousUser: "← Previous",
  nextUser: "Next →",
  discoverTitle: "Find your next game",
  discoverLead: (max: number) =>
    `Search for and pick up to ${max} games you liked, and the two-tower model will ` +
    "recommend others. You can also see a player's recommendations by their number, above " +
    "(1 = most active).",
  searchLoading: "Loading the game catalog…",
  searchPlaceholder: (size: number) => `Search ${size.toLocaleString("en")} games…`,
  searchFull: (max: number) => `${max} games at most: remove one to pick another`,
  searchLabel: "Search games",
  searchUnavailable: "Search is not available",
  noResults: "No results",
  picked: (n: number, max: number) => `${n} of ${max} games picked`,
  pickOrder: " · 1 = most recent. Drag them (or use the arrow keys) to change the order.",
  pickLabel: (position: number, name: string) => `${position}. ${name}`,
  noneChosen: "You haven't picked any games yet.",
  remove: (name: string) => `Remove ${name}`,
  recommend: "Recommend",
  clear: "Clear",
  startHint: "Pick at least one game to see recommendations.",
  unknownPicked: (n: number) =>
    n === 1
      ? "1 picked game is not in the model's catalog."
      : `${n} picked games are not in the model's catalog.`,
  noRecommendations: "No recommendations.",
  likedTitle: "Their latest favorite games",
  likedHint: "This player's most recent positive reviews: the model starts from them.",
  recommendationsTitle: "Recommendations",
  explainedBy: (model: string) => `model: ${model}`,
  summaryReranked: (when: string, models: string) =>
    `Personalized for this player (${models}), reordered and explained by an LLM (${when}).`,
  summaryPersonalized: (when: string, models: string) =>
    `Personalized for this player (${models}, ${when}).`,
  summaryExplained: (when: string, models: string) =>
    `Personalized for this player (${models}), explained by an LLM (${when}).`,
  summaryOnline: (models: string) => `Computed now from the games you picked (${models}).`,
  modelsLabel: (retriever: string, ranker?: string) =>
    ranker ? `retrieval: ${retriever}, ranking: ${ranker}` : `two-tower model: ${retriever}`,
  summaryFallback: (when: string) =>
    `No recommendations for this player yet: showing the most popular games (${when}).`,
  summaryPopular: (when: string) => `Most popular games in recent reviews (${when}).`,
  free: "Free",
  released: "Released",
  price: "Price",
  developer: "Developer",
  publisher: "Publisher",
  genres: "Genres",
  viewOnSteam: "View on Steam ↗",
  errors: {
    400: "The request is not valid.",
    404: "What you were looking for was not found.",
    503: "The model is not available yet. Try again later.",
    other: (status: number) => `Server error (HTTP ${status}). Try again.`,
    network: "Could not connect to the server.",
    unknown: "Something went wrong. Try again.",
    searchIndex: "The search index has an unexpected format.",
  },
  languageSwitch: "Español",
  description: "Steam game recommendations from a two-tower model, a ranker and LLM explanations.",
};

export const MESSAGES: Record<Locale, Messages> = { en, es };
export const t: Messages = MESSAGES[LOCALE];

// Steam's genre names (English in the data) -> Spanish, as the Spanish Steam store shows them.
const GENRES: Record<string, string> = {
  Action: "Acción",
  Adventure: "Aventura",
  Casual: "Casual",
  Indie: "Indie",
  "Massively Multiplayer": "Multijugador masivo",
  RPG: "Rol",
  Racing: "Carreras",
  Simulation: "Simulación",
  Sports: "Deportes",
  Strategy: "Estrategia",
  "Free To Play": "Gratuito",
  "Free to Play": "Gratuito",
  "Early Access": "Acceso anticipado",
  Violent: "Violento",
  Gore: "Gore",
  Education: "Educación",
  Utilities: "Utilidades",
  "Animation & Modeling": "Animación y modelado",
  "Audio Production": "Producción de audio",
  "Design & Illustration": "Diseño e ilustración",
  "Game Development": "Desarrollo de videojuegos",
  "Photo Editing": "Edición de fotos",
  "Software Training": "Formación de software",
  "Video Production": "Producción de vídeo",
  "Web Publishing": "Publicación web",
  Accounting: "Contabilidad",
  Documentary: "Documental",
  Episodic: "Episódico",
  Movie: "Película",
  Short: "Corto",
  Tutorial: "Tutorial",
  "360 Video": "Vídeo 360",
};

export function genreName(genre: string): string {
  return LOCALE === "es" ? (GENRES[genre] ?? genre) : genre;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(LOCALE, {
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}
