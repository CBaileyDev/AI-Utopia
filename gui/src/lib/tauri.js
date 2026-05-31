/* Native-window bridge for the custom titlebar.
 *
 * The React app draws its own window chrome (titlebar + minimize/maximize/close
 * buttons), and the OS window is borderless (`decorations: false` in
 * tauri.conf.json). These helpers wire the buttons to the real OS window via the
 * Tauri v2 core window API.
 *
 * Everything is guarded by `isTauri()` so the exact same build still runs in a
 * plain browser (`npm run dev` without Tauri, or `vite preview`): outside Tauri
 * the calls become harmless no-ops instead of throwing.
 *
 * `window.__TAURI_INTERNALS__` is the v2 runtime sentinel — present only when the
 * page is hosted inside a Tauri webview. The static `import` below is always
 * safe (the module just loads); only the *calls* would throw outside Tauri,
 * which is why we feature-detect before calling rather than around the import.
 */
import { getCurrentWindow } from '@tauri-apps/api/window';

export function isTauri() {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

export async function winMinimize() {
  if (!isTauri()) return;
  try {
    await getCurrentWindow().minimize();
  } catch (e) {
    console.warn('[tauri] minimize failed', e);
  }
}

export async function winToggleMaximize() {
  if (!isTauri()) return;
  try {
    await getCurrentWindow().toggleMaximize();
  } catch (e) {
    console.warn('[tauri] toggleMaximize failed', e);
  }
}

export async function winClose() {
  if (!isTauri()) return;
  try {
    await getCurrentWindow().close();
  } catch (e) {
    console.warn('[tauri] close failed', e);
  }
}
