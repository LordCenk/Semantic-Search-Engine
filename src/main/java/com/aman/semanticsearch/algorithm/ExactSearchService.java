package com.aman.semanticsearch.algorithm;

import com.aman.semanticsearch.search.hnsw.HnswSearchResult;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Brute-Force (Exact) Nearest Neighbor Search implementation.
 *
 * Used as a ground-truth baseline to:
 * 1. Validate the mathematical correctness of HNSW vector similarity.
 * 2. Calculate Search Recall@K (percentage of true exact nearest neighbors found by HNSW).
 * 3. Benchmark latency and throughput trade-offs between exact linear scan and approximate graph navigation.
 */
@Service
public class ExactSearchService {

    // Store of all vectors keyed by document ID
    private final Map<Long, float[]> vectorStore = new ConcurrentHashMap<>();

    /**
     * Indexes or updates a vector in the exact store.
     */
    public void indexVector(long id, float[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector cannot be null");
        }
        vectorStore.put(id, vector);
    }

    /**
     * Removes a vector from the exact store.
     */
    public void removeVector(long id) {
        vectorStore.remove(id);
    }

    /**
     * Clears all vectors in the store.
     */
    public void clear() {
        vectorStore.clear();
    }

    /**
     * Returns the number of indexed vectors.
     */
    public int size() {
        return vectorStore.size();
    }

    /**
     * Performs exact brute-force search using a bounded max-heap of size topK.
     * Time Complexity: O(N * D + N * log(K)) where N is document count, D is dimension, and K is topK.
     * Space Complexity: O(K) for the heap.
     *
     * @param queryVector high-dimensional query vector
     * @param topK number of nearest neighbors to retrieve
     * @return exact top-K results sorted by distance ascending (closest first)
     */
    public List<HnswSearchResult> search(float[] queryVector, int topK) {
        if (queryVector == null || topK <= 0 || vectorStore.isEmpty()) {
            return Collections.emptyList();
        }

        int k = Math.min(topK, vectorStore.size());

        // Max-heap storing pairs, ordered so the FARTHEST neighbor is at the top of the heap.
        // This allows dropping candidates worse than the current worst in O(log K) without full sort.
        PriorityQueue<HnswSearchResult> maxHeap = new PriorityQueue<>(k, (a, b) -> Double.compare(b.getDistance(), a.getDistance()));

        for (Map.Entry<Long, float[]> entry : vectorStore.entrySet()) {
            long docId = entry.getKey();
            float[] targetVector = entry.getValue();

            double distance = VectorSimilarity.cosineDistance(queryVector, targetVector);
            double similarity = 1.0 - distance;

            HnswSearchResult candidate = new HnswSearchResult(docId, distance, similarity);

            if (maxHeap.size() < k) {
                maxHeap.offer(candidate);
            } else if (distance < maxHeap.peek().getDistance()) {
                maxHeap.poll();
                maxHeap.offer(candidate);
            }
        }

        // Extract from max-heap and sort ascending by distance (closest first)
        List<HnswSearchResult> results = new ArrayList<>(maxHeap.size());
        while (!maxHeap.isEmpty()) {
            results.add(maxHeap.poll());
        }
        Collections.reverse(results);
        return results;
    }

    /**
     * Brute-force search using full array sort (O(N * log N)).
     * Provided for benchmarking heap-based vs full-sort brute force.
     */
    public List<HnswSearchResult> searchWithFullSort(float[] queryVector, int topK) {
        if (queryVector == null || topK <= 0 || vectorStore.isEmpty()) {
            return Collections.emptyList();
        }

        List<HnswSearchResult> all = new ArrayList<>(vectorStore.size());
        for (Map.Entry<Long, float[]> entry : vectorStore.entrySet()) {
            double distance = VectorSimilarity.cosineDistance(queryVector, entry.getValue());
            all.add(new HnswSearchResult(entry.getKey(), distance, 1.0 - distance));
        }

        Collections.sort(all);
        int limit = Math.min(topK, all.size());
        return all.subList(0, limit);
    }

    /**
     * Calculates the Recall@K metric comparing approximate results to ground-truth exact results.
     * Recall@K = | approximate_results ∩ exact_results | / K
     *
     * @param approximateResults results returned by HNSW index
     * @param exactResults ground-truth results returned by ExactSearchService
     * @return recall fraction between 0.0 (no overlap) and 1.0 (perfect overlap)
     */
    public static double calculateRecall(List<HnswSearchResult> approximateResults, List<HnswSearchResult> exactResults) {
        if (exactResults.isEmpty()) {
            return 1.0;
        }

        Set<Long> exactIds = new HashSet<>();
        for (HnswSearchResult r : exactResults) {
            exactIds.add(r.getNodeId());
        }

        int matches = 0;
        for (HnswSearchResult r : approximateResults) {
            if (exactIds.contains(r.getNodeId())) {
                matches++;
            }
        }

        return (double) matches / (double) exactResults.size();
    }
}
