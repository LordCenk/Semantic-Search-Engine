package com.aman.semanticsearch.dto;

import java.util.List;

public class SearchResponse {

    private String query;
    private String searchMethod;
    private double executionTimeMs;
    private int totalResults;
    private List<SearchResultResponse> results;

    public SearchResponse() {
    }

    public SearchResponse(String query, String searchMethod, double executionTimeMs, int totalResults, List<SearchResultResponse> results) {
        this.query = query;
        this.searchMethod = searchMethod;
        this.executionTimeMs = executionTimeMs;
        this.totalResults = totalResults;
        this.results = results;
    }

    public String getQuery() {
        return query;
    }

    public void setQuery(String query) {
        this.query = query;
    }

    public String getSearchMethod() {
        return searchMethod;
    }

    public void setSearchMethod(String searchMethod) {
        this.searchMethod = searchMethod;
    }

    public double getExecutionTimeMs() {
        return executionTimeMs;
    }

    public void setExecutionTimeMs(double executionTimeMs) {
        this.executionTimeMs = executionTimeMs;
    }

    public int getTotalResults() {
        return totalResults;
    }

    public void setTotalResults(int totalResults) {
        this.totalResults = totalResults;
    }

    public List<SearchResultResponse> getResults() {
        return results;
    }

    public void setResults(List<SearchResultResponse> results) {
        this.results = results;
    }
}
