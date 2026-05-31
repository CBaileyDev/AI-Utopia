import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import { isTauri } from './lib/tauri.js';
import './styles.css';

// Native feel: suppress the default right-click context menu when running inside
// the Tauri desktop shell. Left intact in a plain browser so dev tooling works.
if (isTauri()) {
  window.addEventListener('contextmenu', (e) => e.preventDefault());
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
