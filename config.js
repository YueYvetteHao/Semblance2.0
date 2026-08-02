// Frontend config for the two-tab recommender.
//
// Tab A (Cell Line Lookup) needs nothing here — it reads static JSON from DATA_BASE and works
// even when the backend is asleep, cold, or not yet deployed.
//
// Tab B (Signature Match) calls BACKEND_URL/semblance on Google Cloud Run. The URL is chosen
// automatically from where the page is being served, so this file does not need editing between
// local development and a push to GitHub Pages:
//
//   localhost / 127.0.0.1 / ::1  -> LOCAL_BACKEND    (uvicorn on your machine)
//   anywhere else                -> CLOUD_RUN_BACKEND
//
// Set CLOUD_RUN_BACKEND once, after `deploy.sh` prints the service URL. No trailing slash.
// Leave it "" to ship Tab A only — Tab B then shows a friendly "coming soon" note instead of
// failing, which is the right behaviour if the backend is ever taken down.
//
// Reminder: the backend's ALLOWED_ORIGINS must list the *origin* serving this page — scheme and
// host only, never a path. For GitHub Pages that is https://yueyvettehao.github.io, NOT
// https://yueyvettehao.github.io/Semblance2.0/. Opening index.html as a file:// URL sends
// "Origin: null" and can never pass CORS, so use `python3 -m http.server 8000` for local dev.
(function () {
  var CLOUD_RUN_BACKEND = "https://semblance-646267029689.us-central1.run.app";
  var LOCAL_BACKEND     = "http://localhost:8080";  // uvicorn main:app --port 8080

  var h = location.hostname;
  var isLocal = h === "localhost" || h === "127.0.0.1" || h === "::1" || h === "[::1]";

  // ?backend=... overrides everything, so a deployed page can be pointed at a local or staging
  // engine for one session without a rebuild. Handy for debugging CORS against the live site.
  var override = new URLSearchParams(location.search).get("backend");

  window.APP_CONFIG = {
    BACKEND_URL: (override || (isLocal ? LOCAL_BACKEND : CLOUD_RUN_BACKEND) || "").replace(/\/+$/, ""),
    DATA_BASE: "./data",   // static Mode-A artifacts, served by GitHub Pages
  };
})();
