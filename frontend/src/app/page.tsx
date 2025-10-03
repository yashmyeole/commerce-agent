"use client";

import { useState } from "react";

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSearch(e) {
    e?.preventDefault();
    setError(null);
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      if (!res.ok) throw new Error("Search failed");
      const data = await res.json();
      setResults(data.results || []);
    } catch (err) {
      console.error(err);
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-2xl font-semibold mb-4">Commerce Agent — Text Search (Demo)</h1>

        <form onSubmit={handleSearch} className="mb-6">
          <div className="flex gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Type something like: 'sports t-shirt for running'"
              className="flex-1 p-3 border rounded-md"
            />
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md disabled:opacity-50"
              disabled={loading}
            >
              {loading ? "Searching..." : "Search"}
            </button>
          </div>
        </form>

        {error && <div className="text-red-600 mb-4">Error: {error}</div>}

        <div className="space-y-4">
          {results.length === 0 && !loading && (
            <div className="text-gray-600">No results yet — try a query.</div>
          )}

          {results.map((p) => (
            <div key={p.id} className="p-4 border rounded-md bg-white">
              <div className="flex items-start gap-4">
                <div className="flex-1">
                  <div className="flex justify-between items-start">
                    <h2 className="text-lg font-medium">{p.title}</h2>
                    <div className="text-sm text-gray-500">${p.price ?? "—"}</div>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{p.description}</p>
                  <div className="text-xs text-gray-500 mt-2">Score: {p.score?.toFixed(3)}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
