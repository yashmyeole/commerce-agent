"use client";

import { useState } from "react";

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchType, setSearchType] = useState("text"); // "text" or "image"
  const [file, setFile] = useState(null);

  const backendOrigin = "http://127.0.0.1:8000";

  async function handleTextSearch(e) {
    e?.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${backendOrigin}/api/search/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      if (!res.ok) throw new Error("Text search failed");
      const data = await res.json();
      setResults(data.results || []);
    } catch (err) {
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function handleImageSearch(e) {
    e?.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${backendOrigin}/api/search/image?top_k=5`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Image search failed");
      const data = await res.json();
      setResults(data.results || []);
    } catch (err) {
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-2xl font-semibold mb-4">Commerce Agent Demo</h1>

        {/* Search type toggle */}
        <div className="flex gap-2 mb-4">
          <button
            className={`px-4 py-2 rounded-md ${
              searchType === "text" ? "bg-blue-600 text-white" : "bg-gray-200"
            }`}
            onClick={() => setSearchType("text")}
          >
            Text Search
          </button>
          <button
            className={`px-4 py-2 rounded-md ${
              searchType === "image" ? "bg-blue-600 text-white" : "bg-gray-200"
            }`}
            onClick={() => setSearchType("image")}
          >
            Image Search
          </button>
        </div>

        {/* Search form */}
        {searchType === "text" ? (
          <form onSubmit={handleTextSearch} className="mb-6 flex gap-2">
            <input
              className="flex-1 p-3 border rounded-md"
              placeholder="Type query, e.g., 'sports t-shirt for running'"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md disabled:opacity-50"
              disabled={loading}
            >
              {loading ? "Searching..." : "Search"}
            </button>
          </form>
        ) : (
          <form onSubmit={handleImageSearch} className="mb-6 flex gap-2 items-center">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files[0])}
              className="p-2 border rounded-md"
            />
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md disabled:opacity-50"
              disabled={loading || !file}
            >
              {loading ? "Searching..." : "Search"}
            </button>
          </form>
        )}

        {error && <div className="text-red-600 mb-4">{error}</div>}

        {/* Results */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {results.map((p) => (
            <div key={p.id} className="p-4 border rounded-md bg-white">
              {p.image_url && (
                <img
                  src={p.image_url.startsWith("/static") ? backendOrigin + p.image_url : p.image_url}
                  alt={p.title}
                  className="w-full h-40 object-contain mb-2"
                />
              )}
              <h2 className="text-lg font-medium">{p.title}</h2>
              {p.description && <p className="text-sm text-gray-600">{p.description}</p>}
              {p.price && <div className="text-sm font-semibold mt-1">${p.price}</div>}
              <div className="text-xs text-gray-500 mt-1">Score: {p.score?.toFixed(3)}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
