package com.aman.semanticsearch.search.hnsw;

import java.util.*;

/**
 * Represents a single node in the multi-layer HNSW graph.
 *
 * Each node corresponds to a document and holds its high-dimensional embedding vector,
 * the top layer assigned to it during insertion, and its neighbor links across layers [0..level].
 */
public class HnswNode {

    private final long id;
    private final float[] vector;
    private final int level;

    // Adjacency sets for each layer: index represents the layer number (0 to level).
    // LinkedHashSet ensures no duplicate neighbor connections and preserves insertion order.
    private final List<Set<Long>> connectionsByLayer;

    public HnswNode(long id, float[] vector, int level) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("Vector cannot be null or empty");
        }
        if (level < 0) {
            throw new IllegalArgumentException("Node level must be non-negative");
        }
        this.id = id;
        this.vector = vector;
        this.level = level;
        this.connectionsByLayer = new ArrayList<>(level + 1);

        for (int i = 0; i <= level; i++) {
            this.connectionsByLayer.add(new LinkedHashSet<>());
        }
    }

    public long getId() {
        return id;
    }

    public float[] getVector() {
        return vector;
    }

    public int getLevel() {
        return level;
    }

    /**
     * Returns the neighbor IDs connected to this node at a given layer.
     * Note: Access is coordinated via HnswIndex's ReentrantReadWriteLock.
     */
    public Set<Long> getNeighbors(int layer) {
        if (layer < 0 || layer > level) {
            return Collections.emptySet();
        }
        return connectionsByLayer.get(layer);
    }

    /**
     * Connects this node to another neighbor node at the specified layer.
     * Prevents self-loops.
     */
    public void addNeighbor(int layer, long neighborId) {
        if (neighborId == this.id) {
            return; // Disallow self-connection
        }
        if (layer >= 0 && layer <= level) {
            connectionsByLayer.get(layer).add(neighborId);
        }
    }

    /**
     * Removes a neighbor connection at a specific layer.
     */
    public void removeNeighbor(int layer, long neighborId) {
        if (layer >= 0 && layer <= level) {
            connectionsByLayer.get(layer).remove(neighborId);
        }
    }

    /**
     * Replaces the neighbor set at a given layer with a pruned set.
     */
    public void setNeighbors(int layer, Collection<Long> newNeighbors) {
        if (layer >= 0 && layer <= level) {
            Set<Long> layerConnections = connectionsByLayer.get(layer);
            layerConnections.clear();
            for (Long neighborId : newNeighbors) {
                if (neighborId != this.id) {
                    layerConnections.add(neighborId);
                }
            }
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof HnswNode hnswNode)) return false;
        return id == hnswNode.id;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(id);
    }

    @Override
    public String toString() {
        return "HnswNode{" +
                "id=" + id +
                ", level=" + level +
                ", vectorDim=" + vector.length +
                '}';
    }
}
