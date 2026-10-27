package com.aman.semanticsearch.dto;

public class SearchResultResponse {

    private Long documentId;
    private String title;
    private String excerpt;
    private double similarityScore;
    private double distance;

    public SearchResultResponse() {
    }

    public SearchResultResponse(Long documentId, String title, String excerpt, double similarityScore, double distance) {
        this.documentId = documentId;
        this.title = title;
        this.excerpt = excerpt;
        this.similarityScore = similarityScore;
        this.distance = distance;
    }

    public Long getDocumentId() {
        return documentId;
    }

    public void setDocumentId(Long documentId) {
        this.documentId = documentId;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getExcerpt() {
        return excerpt;
    }

    public void setExcerpt(String excerpt) {
        this.excerpt = excerpt;
    }

    public double getSimilarityScore() {
        return similarityScore;
    }

    public void setSimilarityScore(double similarityScore) {
        this.similarityScore = similarityScore;
    }

    public double getDistance() {
        return distance;
    }

    public void setDistance(double distance) {
        this.distance = distance;
    }
}
