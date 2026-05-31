//! AI Utopia — Control Center (Tauri v2 desktop shell).
//!
//! This app wraps the existing Vite + React GUI in a native Windows window.
//! There is no Rust ↔ JS command surface yet: the window is borderless
//! (`decorations: false`) and the React titlebar drives the OS window purely
//! through the core JS window API (`@tauri-apps/api/window`), which is gated by
//! the `core:window:*` permissions in `capabilities/default.json`. No custom
//! `#[tauri::command]`s are required for minimize / maximize / close / drag.

pub fn run() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running AI Utopia");
}
