// Viewer request of the app (default) behavior: paths without a file extension are app
// routes (/, /u/123): serve index.html and let the browser route them.
function handler(event) {
  var request = event.request;
  var last = request.uri.split("/").pop();
  if (last.indexOf(".") === -1) {
    request.uri = "/index.html";
  }
  return request;
}
