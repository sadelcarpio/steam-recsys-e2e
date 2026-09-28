// Viewer request of the /api/* and /data/* behaviors: drop the first path segment, so the
// origin sees its own paths (/api/popular -> /popular, /data/games.json -> /games.json).
function handler(event) {
  var request = event.request;
  var rest = request.uri.replace(/^\/[^\/]+/, "");
  request.uri = rest === "" ? "/" : rest;
  return request;
}
