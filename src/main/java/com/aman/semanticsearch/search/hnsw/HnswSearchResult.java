package com.aman.semanticsearch.search.hnsw;

import java.util.Objects;

/**
 * Encapsulates a nearest-neighbor search result containing the document/node ID,
 * calculated distance, and cosine similarity score.
 */
public class HnswSearchResult implements Comparable<HnswSearchResult> {

    private final long nodeId;
    private final double distance;
    private final double similarity;

    public HnswSearchResult(long nodeId, double distance, double similarity) {
        this.nodeId = nodeId;
        this.distance = distance;
        this.similarity = similarity;
    }

    public long getNodeId() {
        return nodeId;
    }

    public double getDistance() {
        return distance;
    }

    public double getSimilarity() {
        return similarity;
    }

    /**
     * Natural ordering is by distance ascending (nearest neighbor first).
     */
    @Override
    public int compareTo(HnswSearchResult other) {
        int cmp = Double.compare(this.distance, other.distance);
        if (cmp != 0) {
            return cmp;
        }
        return Long.compare(this.nodeId, other.nodeId);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof HnswSearchResult that)) return false;
        return nodeId == that.nodeId && Double.compare(that.distance, distance) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(nodeId, distance);
    }

    @Override
    public String toString() {
        return "HnswSearchResult{" +
                "nodeId=" + nodeId +
                ", distance=" + String.format("%.4f", distance) +
                ", similarity=" + String.format("%.4f", similarity) +
                '}';
    }
}
