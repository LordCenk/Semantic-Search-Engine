package com.aman.semanticsearch.search.hnsw;

import com.aman.semanticsearch.algorithm.VectorSimilarity;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Custom Java implementation of the Hierarchical Navigable Small World (HNSW) graph algorithm.
 *
 * References:
 * - Yu. A. Malkov, D. A. Yashunin: "Efficient and robust approximate nearest neighbor search using
 *   Hierarchical Navigable Small World graphs", IEEE TPAMI 2018.
 *
 * Algorithmic Principles:
 * 1. Multi-Layer Hierarchy:
 *    Like a Skip-List for high-dimensional metric spaces.
 *    - Top layers contain sparse nodes with long-range links for rapid coarse routing (O(log N)).
 *    - Bottom layers contain denser connections for fine-grained local navigation.
 * 2. Probabilistic Layer Assignment:
 *    Nodes are randomly assigned a maximum level `l = floor(-ln(uniform(0,1)) * mL)`.
 *    The probability of a node existing at layer l decays exponentially as (1/M)^l.
 * 3. Greedy Routing:
 *    At upper layers, search traverses greedily to the nearest neighbor until a local minimum is reached.
 * 4. Priority Queue Candidate Exploration (Beam Search):
 *    At lower layers, a dynamic candidate list of size `ef` is explored using a min-heap (candidates to visit)
 *    and a max-heap (current closest results, bounded to `ef`).
 * 5. Degree Pruning:
 *    Each node retains at most M connections (2*M at layer 0) to bound memory and search complexity.
 */
public class HnswIndex {

    private final HnswConfig config;
    private final Random random;
    private final ReentrantReadWriteLock rwLock = new ReentrantReadWriteLock(true);

    // Map storing all nodes in the index keyed by document ID
    private final Map<Long, HnswNode> nodes = new ConcurrentHashMap<>();

    // Entry point node for graph traversal (node with the highest level in the graph)
    private volatile HnswNode entryPoint;

    // Highest layer currently present in the graph (-1 when empty)
    private volatile int maxLayer = -1;

    // Expected vector dimension (validated upon insertion)
    private volatile int dimension = -1;

    public HnswIndex() {
        this(new HnswConfig(), new Random());
    }

    public HnswIndex(HnswConfig config) {
        this(config, new Random());
    }

    public HnswIndex(HnswConfig config, Random random) {
        if (config == null) {
            throw new IllegalArgumentException("HnswConfig cannot be null");
        }
        this.config = config;
        this.random = random != null ? random : new Random();
    }

    /**
     * Inserts a vector into the HNSW graph index.
     *
     * @param id unique document identifier
     * @param vector high-dimensional embedding vector
     */
    public void insert(long id, float[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector cannot be null");
        }
        VectorSimilarity.validateVector(vector, dimension > 0 ? dimension : vector.length);

        rwLock.writeLock().lock();
        try {
            // Validate and record dimension on first insertion
            if (dimension == -1) {
                dimension = vector.length;
            }

            // If node already exists, remove it first to ensure clean topology
            if (nodes.containsKey(id)) {
                deleteInternal(id);
            }

            // 1. Probabilistic layer selection: l ~ -ln(uniform) * mL
            int nodeLevel = sampleRandomLevel();
            HnswNode newNode = new HnswNode(id, vector, nodeLevel);
            nodes.put(id, newNode);

            // Base Case: First node in the graph becomes the entry point
            if (entryPoint == null) {
                entryPoint = newNode;
                maxLayer = nodeLevel;
                return;
            }

            HnswNode currObj = entryPoint;
            int currentMaxLayer = this.maxLayer;

            // 2. Greedy search through upper layers [currentMaxLayer .. nodeLevel + 1]
            // At these layers, we only need to find the single closest entry point for the next layer down.
            for (int layer = currentMaxLayer; layer > nodeLevel; layer--) {
                currObj = findGreedyNearest(currObj, vector, layer);
            }

            // 3. Search and connect through layers [min(currentMaxLayer, nodeLevel) .. 0]
            // In these layers, the new node will establish bidirectional connections with up to M neighbors.
            for (int layer = Math.min(currentMaxLayer, nodeLevel); layer >= 0; layer--) {
                // Explore candidates at this layer with beam width efConstruction
                PriorityQueue<DistPair> candidateSet = searchLayer(vector, currObj, config.getEfConstruction(), layer);

                // Select closest M neighbors from the candidate set (M_max0 at ground layer 0)
                int maxM = (layer == 0) ? config.getMMax0() : config.getM();
                List<HnswNode> selectedNeighbors = selectNeighbors(candidateSet, maxM);

                // Create bidirectional connections between newNode and each selected neighbor
                for (HnswNode neighbor : selectedNeighbors) {
                    newNode.addNeighbor(layer, neighbor.getId());
                    neighbor.addNeighbor(layer, newNode.getId());

                    // Prune neighbor's connections if exceeding max allowed degree
                    shrinkConnectionsIfNeeded(neighbor, layer, maxM);
                }

                // Next lower layer starts traversal from the closest candidate found in this layer
                if (!candidateSet.isEmpty()) {
                    currObj = candidateSet.peek().node;
                }
            }

            // 4. If the new node was assigned a level higher than currentMaxLayer, it becomes the new global entry point
            if (nodeLevel > currentMaxLayer) {
                this.maxLayer = nodeLevel;
                this.entryPoint = newNode;
            }
        } finally {
            rwLock.writeLock().unlock();
        }
    }

    /**
     * Approximate Nearest Neighbor Search using HNSW.
     *
     * @param queryVector high-dimensional query vector
     * @param topK number of nearest neighbors to retrieve
     * @return ranked list of search results sorted by distance ascending (closest first)
     */
    public List<HnswSearchResult> search(float[] queryVector, int topK) {
        if (queryVector == null) {
            throw new IllegalArgumentException("Query vector cannot be null");
        }
        if (topK <= 0) {
            return Collections.emptyList();
        }

        rwLock.readLock().lock();
        try {
            if (entryPoint == null || nodes.isEmpty()) {
                return Collections.emptyList();
            }

            VectorSimilarity.validateVector(queryVector, dimension);

            HnswNode currObj = entryPoint;
            int currentMaxLayer = this.maxLayer;

            // 1. Fast greedy navigation through upper layers [currentMaxLayer .. 1]
            for (int layer = currentMaxLayer; layer > 0; layer--) {
                currObj = findGreedyNearest(currObj, queryVector, layer);
            }

            // 2. Candidate exploration at layer 0 with beam width efSearch
            int ef = Math.max(config.getEfSearch(), topK);
            PriorityQueue<DistPair> candidates = searchLayer(queryVector, currObj, ef, 0);

            // 3. Extract top-K nearest results sorted by distance ascending
            List<HnswSearchResult> results = new ArrayList<>();
            while (!candidates.isEmpty() && results.size() < topK) {
                DistPair pair = candidates.poll();
                double similarity = 1.0 - pair.distance;
                results.add(new HnswSearchResult(pair.node.getId(), pair.distance, similarity));
            }

            return results;
        } finally {
            rwLock.readLock().unlock();
        }
    }

    /**
     * Greedy routing at a single layer to find the single local nearest neighbor.
     * Continues stepping to closer neighbors until no neighbor is closer to query than current node.
     */
    private HnswNode findGreedyNearest(HnswNode startNode, float[] query, int layer) {
        HnswNode current = startNode;
        double currentDist = VectorSimilarity.cosineDistance(query, current.getVector());

        boolean changed;
        do {
            changed = false;
            for (Long neighborId : current.getNeighbors(layer)) {
                HnswNode neighbor = nodes.get(neighborId);
                if (neighbor == null) continue;

                double d = VectorSimilarity.cosineDistance(query, neighbor.getVector());
                if (d < currentDist) {
                    currentDist = d;
                    current = neighbor;
                    changed = true;
                }
            }
        } while (changed);

        return current;
    }

    /**
     * SEARCH-LAYER algorithm (Algorithm 2 in Malkov & Yashunin).
     *
     * Explores graph layer using two priority queues:
     * - `candidates` (Min-Heap by distance): nodes to visit next (closest candidate is polled first).
     * - `w` (Max-Heap by distance): current best nearest neighbors found so far, bounded to size `ef`.
     *
     * @param query query vector
     * @param entry entry point node for this layer
     * @param ef exploration breadth factor
     * @param layer layer index
     * @return candidate nearest neighbors sorted by distance ascending (closest first)
     */
    private PriorityQueue<DistPair> searchLayer(float[] query, HnswNode entry, int ef, int layer) {
        Set<Long> visited = new HashSet<>();

        // Min-heap: candidate pool ordered ascending by distance (nearest candidate polled first)
        PriorityQueue<DistPair> candidates = new PriorityQueue<>(Comparator.comparingDouble(p -> p.distance));

        // Max-heap: tracks the closest `ef` nodes discovered so far (farthest node at top for bounding)
        PriorityQueue<DistPair> w = new PriorityQueue<>((a, b) -> Double.compare(b.distance, a.distance));

        double entryDist = VectorSimilarity.cosineDistance(query, entry.getVector());
        DistPair entryPair = new DistPair(entry, entryDist);

        visited.add(entry.getId());
        candidates.offer(entryPair);
        w.offer(entryPair);

        while (!candidates.isEmpty()) {
            DistPair c = candidates.poll();
            DistPair farthestInW = w.peek();

            // Early exit condition: if the closest candidate in the queue is farther than
            // the worst result in our bounded result set w, no further exploration can improve w.
            if (farthestInW != null && c.distance > farthestInW.distance && w.size() >= ef) {
                break;
            }

            for (Long neighborId : c.node.getNeighbors(layer)) {
                if (visited.add(neighborId)) {
                    HnswNode neighbor = nodes.get(neighborId);
                    if (neighbor == null) continue;

                    double d = VectorSimilarity.cosineDistance(query, neighbor.getVector());
                    DistPair e = new DistPair(neighbor, d);

                    farthestInW = w.peek();
                    if (farthestInW == null || d < farthestInW.distance || w.size() < ef) {
                        candidates.offer(e);
                        w.offer(e);

                        if (w.size() > ef) {
                            w.poll(); // Evict the farthest element to maintain bounded beam width ef
                        }
                    }
                }
            }
        }

        // Return results as a min-heap (nearest first)
        PriorityQueue<DistPair> sortedResults = new PriorityQueue<>(Comparator.comparingDouble(p -> p.distance));
        sortedResults.addAll(w);
        return sortedResults;
    }

    /**
     * Selects up to `maxM` closest neighbors from the candidate set.
     */
    private List<HnswNode> selectNeighbors(PriorityQueue<DistPair> candidates, int maxM) {
        List<HnswNode> selected = new ArrayList<>(maxM);
        // Copy to avoid consuming original queue if needed elsewhere
        PriorityQueue<DistPair> copy = new PriorityQueue<>(Comparator.comparingDouble(p -> p.distance));
        copy.addAll(candidates);

        while (!copy.isEmpty() && selected.size() < maxM) {
            selected.add(copy.poll().node);
        }
        return selected;
    }

    /**
     * Shrinks connections of a node at a given layer to at most `maxAllowed` nearest neighbors.
     */
    private void shrinkConnectionsIfNeeded(HnswNode node, int layer, int maxAllowed) {
        Set<Long> neighbors = node.getNeighbors(layer);
        if (neighbors.size() <= maxAllowed) {
            return;
        }

        // Calculate distance from node to each neighbor and select the closest maxAllowed
        PriorityQueue<DistPair> neighborDistances = new PriorityQueue<>(Comparator.comparingDouble(p -> p.distance));
        for (Long nId : neighbors) {
            HnswNode neighbor = nodes.get(nId);
            if (neighbor != null) {
                double dist = VectorSimilarity.cosineDistance(node.getVector(), neighbor.getVector());
                neighborDistances.offer(new DistPair(neighbor, dist));
            }
        }

        List<Long> pruned = new ArrayList<>(maxAllowed);
        while (!neighborDistances.isEmpty() && pruned.size() < maxAllowed) {
            pruned.add(neighborDistances.poll().node.getId());
        }

        node.setNeighbors(layer, pruned);
    }

    /**
     * Samples a random level for a new node using exponential decay distribution.
     * Level = floor(-ln(uniform(0, 1)) * mL)
     */
    private int sampleRandomLevel() {
        double r = random.nextDouble();
        // Prevent log(0.0) which yields -Infinity
        if (r == 0.0) {
            r = 1e-9;
        }
        return (int) (-Math.log(r) * config.getML());
    }

    /**
     * Deletes a document vector from the index.
     *
     * @param id document identifier
     * @return true if document was found and removed, false otherwise
     */
    public boolean delete(long id) {
        rwLock.writeLock().lock();
        try {
            return deleteInternal(id);
        } finally {
            rwLock.writeLock().unlock();
        }
    }

    private boolean deleteInternal(long id) {
        HnswNode removed = nodes.remove(id);
        if (removed == null) {
            return false;
        }

        // Sever all bidirectional connections from neighboring nodes across all layers
        for (int l = 0; l <= removed.getLevel(); l++) {
            for (Long neighborId : removed.getNeighbors(l)) {
                HnswNode neighbor = nodes.get(neighborId);
                if (neighbor != null) {
                    neighbor.removeNeighbor(l, id);
                }
            }
        }

        // If the removed node was the global entry point, select a new entry point
        if (entryPoint != null && entryPoint.getId() == id) {
            if (nodes.isEmpty()) {
                entryPoint = null;
                maxLayer = -1;
                dimension = -1;
            } else {
                // Find existing node with the highest level to serve as entry point
                HnswNode bestNewEntry = null;
                for (HnswNode candidate : nodes.values()) {
                    if (bestNewEntry == null || candidate.getLevel() > bestNewEntry.getLevel()) {
                        bestNewEntry = candidate;
                    }
                }
                this.entryPoint = bestNewEntry;
                this.maxLayer = bestNewEntry != null ? bestNewEntry.getLevel() : -1;
            }
        }

        return true;
    }

    /**
     * Clears all nodes from the index.
     */
    public void clear() {
        rwLock.writeLock().lock();
        try {
            nodes.clear();
            entryPoint = null;
            maxLayer = -1;
            dimension = -1;
        } finally {
            rwLock.writeLock().unlock();
        }
    }

    /**
     * Returns the total count of indexed vectors.
     */
    public int size() {
        return nodes.size();
    }

    /**
     * Returns the current maximum graph layer.
     */
    public int getMaxLayer() {
        return maxLayer;
    }

    /**
     * Returns the vector dimension.
     */
    public int getDimension() {
        return dimension;
    }

    /**
     * Returns configuration used by this index.
     */
    public HnswConfig getConfig() {
        return config;
    }

    /**
     * Returns true if document ID exists in index.
     */
    public boolean contains(long id) {
        return nodes.containsKey(id);
    }

    /**
     * Internal helper representing a node and its computed distance to a query vector.
     */
    private static class DistPair {
        final HnswNode node;
        final double distance;

        DistPair(HnswNode node, double distance) {
            this.node = node;
            this.distance = distance;
        }
    }
}
