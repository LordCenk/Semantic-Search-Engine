package com.aman.semanticsearch.algorithm;

import com.aman.semanticsearch.search.hnsw.HnswSearchResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ExactSearchServiceTest {

    private ExactSearchService exactSearchService;

    @BeforeEach
    void setUp() {
        exactSearchService = new ExactSearchService();
    }

    @Test
    @DisplayName("Empty exact store should return empty search results")
    void testEmptyStore() {
        List<HnswSearchResult> results = exactSearchService.search(new float[]{1.0f, 0.0f}, 5);
        assertTrue(results.isEmpty());
        assertEquals(0, exactSearchService.size());
    }

    @Test
    @DisplayName("Exact search should find closest vector accurately")
    void testExactSearch() {
        exactSearchService.indexVector(1L, new float[]{1.0f, 0.0f, 0.0f});
        exactSearchService.indexVector(2L, new float[]{0.0f, 1.0f, 0.0f});
        exactSearchService.indexVector(3L, new float[]{0.7f, 0.7f, 0.0f});

        float[] query = {1.0f, 0.1f, 0.0f};
        List<HnswSearchResult> results = exactSearchService.search(query, 2);

        assertEquals(2, results.size());
        assertEquals(1L, results.get(0).getNodeId());
        assertEquals(3L, results.get(1).getNodeId());
    }

    @Test
    @DisplayName("Heap-based search and full-sort search should yield identical top-K results")
    void testHeapMatchesFullSort() {
        for (long i = 1; i <= 50; i++) {
            exactSearchService.indexVector(i, new float[]{(float) Math.sin(i), (float) Math.cos(i), (float) i});
        }

        float[] query = {0.5f, 0.5f, 10.0f};
        List<HnswSearchResult> heapResults = exactSearchService.search(query, 10);
        List<HnswSearchResult> sortResults = exactSearchService.searchWithFullSort(query, 10);

        assertEquals(heapResults.size(), sortResults.size());
        for (int i = 0; i < heapResults.size(); i++) {
            assertEquals(heapResults.get(i).getNodeId(), sortResults.get(i).getNodeId());
            assertEquals(heapResults.get(i).getDistance(), sortResults.get(i).getDistance(), 1e-6);
        }
    }

    @Test
    @DisplayName("Recall calculation should evaluate overlap fraction correctly")
    void testRecallCalculation() {
        List<HnswSearchResult> exact = List.of(
                new HnswSearchResult(1L, 0.1, 0.9),
                new HnswSearchResult(2L, 0.2, 0.8),
                new HnswSearchResult(3L, 0.3, 0.7),
                new HnswSearchResult(4L, 0.4, 0.6)
        );

        // 3 out of 4 matches
        List<HnswSearchResult> approx = List.of(
                new HnswSearchResult(1L, 0.1, 0.9),
                new HnswSearchResult(2L, 0.2, 0.8),
                new HnswSearchResult(3L, 0.3, 0.7),
                new HnswSearchResult(99L, 0.5, 0.5)
        );

        double recall = ExactSearchService.calculateRecall(approx, exact);
        assertEquals(0.75, recall, 1e-6);
    }
}
