package com.aman.semanticsearch.search.hnsw;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class HnswIndexTest {

    private HnswConfig config;
    private HnswIndex index;

    @BeforeEach
    void setUp() {
        // Use deterministic random seed for repeatable testing
        config = new HnswConfig(4, 16, 16);
        index = new HnswIndex(config, new Random(42));
    }

    @Test
    @DisplayName("Searching an empty index should safely return an empty result list")
    void testSearchEmptyIndex() {
        float[] query = {1.0f, 0.0f, 0.0f};
        List<HnswSearchResult> results = index.search(query, 5);

        assertNotNull(results);
        assertTrue(results.isEmpty());
        assertEquals(0, index.size());
    }

    @Test
    @DisplayName("Inserting a single vector should return it as nearest neighbor")
    void testInsertSingleVector() {
        float[] vector = {1.0f, 2.0f, 3.0f};
        index.insert(100L, vector);

        assertEquals(1, index.size());
        assertEquals(3, index.getDimension());

        List<HnswSearchResult> results = index.search(vector, 5);
        assertEquals(1, results.size());
        assertEquals(100L, results.get(0).getNodeId());
        assertEquals(0.0, results.get(0).getDistance(), 1e-5);
        assertEquals(1.0, results.get(0).getSimilarity(), 1e-5);
    }

    @Test
    @DisplayName("Index should correctly retrieve nearest neighbors in order of similarity")
    void testRetrieveNearestNeighbors() {
        // Vector 1: close to [1, 0, 0]
        float[] v1 = {0.99f, 0.01f, 0.0f};
        // Vector 2: moderately close to [1, 0, 0]
        float[] v2 = {0.80f, 0.60f, 0.0f};
        // Vector 3: orthogonal to [1, 0, 0]
        float[] v3 = {0.0f, 1.0f, 0.0f};

        index.insert(1L, v1);
        index.insert(2L, v2);
        index.insert(3L, v3);

        float[] query = {1.0f, 0.0f, 0.0f};
        List<HnswSearchResult> results = index.search(query, 3);

        assertEquals(3, results.size());
        assertEquals(1L, results.get(0).getNodeId(), "v1 should be closest");
        assertEquals(2L, results.get(1).getNodeId(), "v2 should be second closest");
        assertEquals(3L, results.get(2).getNodeId(), "v3 should be third closest");

        // Verify distances are monotonically non-decreasing
        assertTrue(results.get(0).getDistance() <= results.get(1).getDistance());
        assertTrue(results.get(1).getDistance() <= results.get(2).getDistance());
    }

    @Test
    @DisplayName("Search should return no more than K results")
    void testTopKLimit() {
        for (long i = 1; i <= 20; i++) {
            index.insert(i, new float[]{(float) i, 1.0f, 0.5f});
        }

        assertEquals(20, index.size());

        List<HnswSearchResult> top3 = index.search(new float[]{1.0f, 1.0f, 0.5f}, 3);
        assertEquals(3, top3.size());

        List<HnswSearchResult> top7 = index.search(new float[]{1.0f, 1.0f, 0.5f}, 7);
        assertEquals(7, top7.size());
    }

    @Test
    @DisplayName("Search results should never contain duplicate document IDs")
    void testNoDuplicateResults() {
        for (long i = 1; i <= 15; i++) {
            index.insert(i, new float[]{(float) Math.cos(i), (float) Math.sin(i), (float) i});
        }

        List<HnswSearchResult> results = index.search(new float[]{1.0f, 0.0f, 0.0f}, 10);
        Set<Long> seen = new HashSet<>();
        for (HnswSearchResult res : results) {
            assertTrue(seen.add(res.getNodeId()), "Duplicate result ID found: " + res.getNodeId());
        }
    }

    @Test
    @DisplayName("Incompatible vector dimension should throw IllegalArgumentException")
    void testIncompatibleDimensions() {
        index.insert(1L, new float[]{1.0f, 2.0f, 3.0f});

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                index.insert(2L, new float[]{1.0f, 2.0f}));
        assertTrue(ex.getMessage().contains("dimension"));

        IllegalArgumentException queryEx = assertThrows(IllegalArgumentException.class, () ->
                index.search(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 5));
        assertTrue(queryEx.getMessage().contains("dimension"));
    }

    @Test
    @DisplayName("Deleted document should immediately disappear from search results")
    void testDocumentDeletion() {
        float[] target = {1.0f, 0.0f, 0.0f};
        float[] other = {0.0f, 1.0f, 0.0f};

        index.insert(10L, target);
        index.insert(20L, other);

        // Before deletion: document 10 is found as top result
        List<HnswSearchResult> before = index.search(target, 2);
        assertEquals(10L, before.get(0).getNodeId());

        // Delete document 10
        boolean deleted = index.delete(10L);
        assertTrue(deleted);
        assertEquals(1, index.size());

        // After deletion: document 10 must not be returned
        List<HnswSearchResult> after = index.search(target, 2);
        assertEquals(1, after.size());
        assertEquals(20L, after.get(0).getNodeId());
    }

    @Test
    @DisplayName("Re-inserting same ID should update its vector and topology correctly")
    void testDocumentUpdate() {
        float[] original = {1.0f, 0.0f, 0.0f};
        index.insert(1L, original);

        float[] opposite = {-1.0f, 0.0f, 0.0f};
        index.insert(1L, opposite); // Overwrite ID 1

        assertEquals(1, index.size());

        // Searching with original should now have distance close to 2.0 (opposite)
        List<HnswSearchResult> res = index.search(original, 1);
        assertEquals(1L, res.get(0).getNodeId());
        assertEquals(2.0, res.get(0).getDistance(), 1e-5);
    }

    @Test
    @DisplayName("Random seed repeatability guarantees identical graph construction")
    void testRepeatableConstruction() {
        HnswIndex indexA = new HnswIndex(new HnswConfig(4, 16, 16), new Random(12345));
        HnswIndex indexB = new HnswIndex(new HnswConfig(4, 16, 16), new Random(12345));

        for (long i = 1; i <= 30; i++) {
            float[] vec = {(float) Math.sin(i), (float) Math.cos(i), (float) (i % 5)};
            indexA.insert(i, vec);
            indexB.insert(i, vec);
        }

        float[] query = {0.5f, 0.5f, 1.0f};
        List<HnswSearchResult> resA = indexA.search(query, 5);
        List<HnswSearchResult> resB = indexB.search(query, 5);

        assertEquals(resA.size(), resB.size());
        for (int i = 0; i < resA.size(); i++) {
            assertEquals(resA.get(i).getNodeId(), resB.get(i).getNodeId());
            assertEquals(resA.get(i).getDistance(), resB.get(i).getDistance(), 1e-6);
        }
    }
}
