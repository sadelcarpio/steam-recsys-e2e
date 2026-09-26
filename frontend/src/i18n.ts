// Every text the app renders itself, in Spanish. Data from the API (game names, descriptions,
// LLM explanations) is shown as is; Steam's genre names are translated when known.

export const LOCALE = "es";

export const t = {
  brand: "🎮 Recomendador de juegos de Steam",
  userIdPlaceholder: "ID de usuario de Steam",
  userIdLabel: "ID de usuario de Steam",
  userIdSubmit: "Ver recomendaciones",
  userIdInvalid: "Un ID de cuenta de Steam: de 1 a 20 dígitos",
  notFound: "Página no encontrada. ",
  goHome: "Volver al inicio",
  loading: "Cargando…",
  loadingRecommendations: "Cargando recomendaciones…",
  close: "Cerrar",
  userTitle: "Recomendaciones para ",
  discoverTitle: "Encuentra tu próximo juego",
  discoverLead: (max: number) =>
    `Busca y elige hasta ${max} juegos que te gustaron, y el modelo de dos torres te ` +
    "recomendará otros. También puedes ver las recomendaciones de un jugador con su ID de " +
    "Steam, arriba.",
  searchLoading: "Cargando el catálogo de juegos…",
  searchPlaceholder: (size: number) => `Buscar entre ${size.toLocaleString(LOCALE)} juegos…`,
  searchFull: (max: number) => `Máximo ${max} juegos: quita uno para elegir otro`,
  searchLabel: "Buscar juegos",
  searchUnavailable: "La búsqueda no está disponible",
  noResults: "Sin resultados",
  picked: (n: number, max: number) => `${n} de ${max} juegos elegidos`,
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
  summaryReranked: (when: string) =>
    `Personalizadas para este jugador, reordenadas y explicadas por un LLM (${when}).`,
  summaryPersonalized: (when: string) =>
    `Personalizadas para este jugador por el modelo de dos torres (${when}).`,
  summaryOnline: (model: string) =>
    `Calculadas ahora a partir de los juegos que elegiste (modelo ${model}).`,
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
};

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
  return GENRES[genre] ?? genre;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(LOCALE, {
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}
