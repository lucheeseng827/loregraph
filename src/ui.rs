//! The embedded canvas SPA (rust-embed), served as the axum fallback.
//!
//! `frontend/` is baked into the binary, so the canvas ships air-gapped with no Node
//! toolchain (the evald/evald precedent). Anything not matched by `/v1/*` or `/healthz`
//! falls through here; unknown routes return `index.html` (SPA routing).

use axum::http::{header, StatusCode, Uri};
use axum::response::{IntoResponse, Response};

#[derive(rust_embed::RustEmbed)]
#[folder = "frontend/"]
struct Assets;

/// Serve a bundled asset, falling back to `index.html` for unknown non-API routes.
pub async fn serve_asset(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };

    if let Some(content) = Assets::get(path) {
        let mime = mime_for(path).to_string();
        return ([(header::CONTENT_TYPE, mime)], content.data.into_owned()).into_response();
    }
    // An asset-like miss (the last path segment has a file extension, e.g. `app.js`,
    // `favicon.ico`) is a real 404 — only extensionless route-like paths fall back to the
    // SPA shell.
    let looks_like_asset = path.rsplit('/').next().map(|s| s.contains('.')).unwrap_or(false);
    if looks_like_asset {
        return (StatusCode::NOT_FOUND, "asset not found").into_response();
    }
    match Assets::get("index.html") {
        Some(content) => (
            [(header::CONTENT_TYPE, "text/html; charset=utf-8".to_string())],
            content.data.into_owned(),
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "asset not found").into_response(),
    }
}

/// Minimal extension → content-type map (avoids the `mime-guess` feature, keeping the
/// dependency surface small for the canvas assets we actually ship).
fn mime_for(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}
