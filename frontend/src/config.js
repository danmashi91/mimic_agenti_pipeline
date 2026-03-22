// frontend/src/config.js
// Automatically switches API URL between local dev and Railway production

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default API_URL;
