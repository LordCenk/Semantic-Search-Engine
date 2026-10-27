package com.aman.semanticsearch.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;

public class SearchRequest {

    @NotBlank(message = "Search query must not be blank")
    private String query;

    @Min(value = 1, message = "topK must be at least 1")
    @Max(value = 100, message = "topK cannot exceed 100")
    private int topK = 5;

    /**
     * Optional search method: "HNSW" (approximate graph search) or "EXACT" (brute-force baseline).
     * Defaults to "HNSW".
     */
    private String method = "HNSW";

    public SearchRequest() {
    }

    public SearchRequest(String query, int topK) {
        this.query = query;
        this.topK = topK;
        this.method = "HNSW";
    }

    public SearchRequest(String query, int topK, String method) {
        this.query = query;
        this.topK = topK;
        this.method = method;
    }

    public String getQuery() {
        return query;
    }

    public void setQuery(String query) {
        this.query = query;
    }

    public int getTopK() {
        return topK;
    }

    public void setTopK(int topK) {
        this.topK = topK;
    }

    public String getMethod() {
        return method;
    }

    public void setMethod(String method) {
        this.method = method;
    }
}
