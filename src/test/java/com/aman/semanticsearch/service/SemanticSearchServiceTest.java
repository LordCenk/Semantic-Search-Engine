package com.aman.semanticsearch.service;

import com.aman.semanticsearch.algorithm.ExactSearchService;
import com.aman.semanticsearch.dto.SearchRequest;
import com.aman.semanticsearch.dto.SearchResponse;
import com.aman.semanticsearch.exception.EmbeddingException;
import com.aman.semanticsearch.model.Document;
import com.aman.semanticsearch.repository.DocumentRepository;
import com.aman.semanticsearch.search.hnsw.HnswIndex;
import com.aman.semanticsearch.search.hnsw.HnswSearchResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SemanticSearchServiceTest {

    @Mock
    private EmbeddingService embeddingService;

    @Mock
    private HnswIndex hnswIndex;

    @Mock
    private ExactSearchService exactSearchService;

    @Mock
    private DocumentRepository documentRepository;

    @Mock
    private IndexManagementService indexManagementService;

    private SemanticSearchService semanticSearchService;

    @BeforeEach
    void setUp() {
        semanticSearchService = new SemanticSearchService(
                embeddingService,
                hnswIndex,
                exactSearchService,
                documentRepository,
                indexManagementService
        );
    }

    @Test
    @DisplayName("Successful semantic search should return populated search response")
    void testSuccessfulSearch() {
        when(indexManagementService.isReady()).thenReturn(true);

        float[] mockQueryVector = new float[]{0.1f, 0.2f};
        when(embeddingService.generateEmbedding("spring security")).thenReturn(mockQueryVector);

        List<HnswSearchResult> hnswResults = List.of(
                new HnswSearchResult(1L, 0.1, 0.9)
        );
        when(hnswIndex.search(mockQueryVector, 5)).thenReturn(hnswResults);

        Document doc = new Document("Spring Security Guide", "Authentication details", mockQueryVector);
        doc.setId(1L);
        when(documentRepository.findAllById(List.of(1L))).thenReturn(List.of(doc));

        SearchRequest req = new SearchRequest("spring security", 5, "HNSW");
        SearchResponse response = semanticSearchService.search(req);

        assertNotNull(response);
        assertEquals("spring security", response.getQuery());
        assertEquals("HNSW", response.getSearchMethod());
        assertEquals(1, response.getResults().size());
        assertEquals(1L, response.getResults().get(0).getDocumentId());
        assertEquals("Spring Security Guide", response.getResults().get(0).getTitle());
    }

    @Test
    @DisplayName("Blank query should throw IllegalArgumentException")
    void testBlankQuery() {
        SearchRequest req = new SearchRequest("   ", 5);
        assertThrows(IllegalArgumentException.class, () -> semanticSearchService.search(req));
    }

    @Test
    @DisplayName("Invalid topK should throw IllegalArgumentException")
    void testInvalidTopK() {
        SearchRequest req = new SearchRequest("query", 0);
        assertThrows(IllegalArgumentException.class, () -> semanticSearchService.search(req));
    }

    @Test
    @DisplayName("Index not ready should throw IllegalStateException")
    void testIndexNotReady() {
        when(indexManagementService.isReady()).thenReturn(false);
        SearchRequest req = new SearchRequest("query", 5);
        assertThrows(IllegalStateException.class, () -> semanticSearchService.search(req));
    }

    @Test
    @DisplayName("Embedding provider failure should propagate EmbeddingException")
    void testEmbeddingFailure() {
        when(indexManagementService.isReady()).thenReturn(true);
        when(embeddingService.generateEmbedding(anyString()))
                .thenThrow(new EmbeddingException("API quota exceeded"));

        SearchRequest req = new SearchRequest("query", 5);
        assertThrows(EmbeddingException.class, () -> semanticSearchService.search(req));
    }

    @Test
    @DisplayName("Empty search results should return empty list gracefully")
    void testEmptyResults() {
        when(indexManagementService.isReady()).thenReturn(true);
        when(embeddingService.generateEmbedding(anyString())).thenReturn(new float[]{0.1f});
        when(hnswIndex.search(any(), anyInt())).thenReturn(Collections.emptyList());

        SearchRequest req = new SearchRequest("query", 5);
        SearchResponse response = semanticSearchService.search(req);

        assertNotNull(response);
        assertEquals(0, response.getTotalResults());
        assertTrue(response.getResults().isEmpty());
    }
}
