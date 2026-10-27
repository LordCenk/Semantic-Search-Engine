package com.aman.semanticsearch.service;

import com.aman.semanticsearch.algorithm.ExactSearchService;
import com.aman.semanticsearch.dto.SearchRequest;
import com.aman.semanticsearch.dto.SearchResponse;
import com.aman.semanticsearch.dto.SearchResultResponse;
import com.aman.semanticsearch.model.Document;
import com.aman.semanticsearch.repository.DocumentRepository;
import com.aman.semanticsearch.search.hnsw.HnswIndex;
import com.aman.semanticsearch.search.hnsw.HnswSearchResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Orchestrates semantic similarity search across user queries, embedding conversion,
 * HNSW graph navigation, and result formatting.
 */
@Service
public class SemanticSearchService {

    private static final Logger log = LoggerFactory.getLogger(SemanticSearchService.class);

    private final EmbeddingService embeddingService;
    private final HnswIndex hnswIndex;
    private final ExactSearchService exactSearchService;
    private final DocumentRepository documentRepository;
    private final IndexManagementService indexManagementService;

    public SemanticSearchService(EmbeddingService embeddingService,
                                 HnswIndex hnswIndex,
                                 ExactSearchService exactSearchService,
                                 DocumentRepository documentRepository,
                                 IndexManagementService indexManagementService) {
        this.embeddingService = embeddingService;
        this.hnswIndex = hnswIndex;
        this.exactSearchService = exactSearchService;
        this.documentRepository = documentRepository;
        this.indexManagementService = indexManagementService;
    }

    /**
     * Executes semantic vector search using either approximate HNSW or exact brute force.
     */
    public SearchResponse search(SearchRequest request) {
        validateRequest(request);

        String method = (request.getMethod() != null && request.getMethod().equalsIgnoreCase("EXACT"))
                ? "EXACT"
                : "HNSW";

        // 1. Convert natural language query into high-dimensional vector
        float[] queryVector = embeddingService.generateEmbedding(request.getQuery());

        // 2. Perform nearest neighbor search and measure execution time
        long startTimeNano = System.nanoTime();
        List<HnswSearchResult> searchResults;

        if ("EXACT".equals(method)) {
            searchResults = exactSearchService.search(queryVector, request.getTopK());
        } else {
            searchResults = hnswIndex.search(queryVector, request.getTopK());
        }
        long durationNano = System.nanoTime() - startTimeNano;
        double durationMs = durationNano / 1_000_000.0;

        // 3. Hydrate matching documents for the result items
        List<SearchResultResponse> items = buildResultItems(searchResults);

        return new SearchResponse(
                request.getQuery(),
                method,
                durationMs,
                items.size(),
                items
        );
    }

    /**
     * Executes both HNSW and Exact searches to compare latency, throughput, and Recall@K.
     */
    public Map<String, Object> compare(SearchRequest request) {
        validateRequest(request);

        float[] queryVector = embeddingService.generateEmbedding(request.getQuery());

        // Run HNSW Search
        long hnswStart = System.nanoTime();
        List<HnswSearchResult> hnswResults = hnswIndex.search(queryVector, request.getTopK());
        double hnswTimeMs = (System.nanoTime() - hnswStart) / 1_000_000.0;

        // Run Exact Brute-Force Search
        long exactStart = System.nanoTime();
        List<HnswSearchResult> exactResults = exactSearchService.search(queryVector, request.getTopK());
        double exactTimeMs = (System.nanoTime() - exactStart) / 1_000_000.0;

        // Calculate Recall@K
        double recall = ExactSearchService.calculateRecall(hnswResults, exactResults);

        Map<String, Object> comparison = new HashMap<>();
        comparison.put("query", request.getQuery());
        comparison.put("topK", request.getTopK());
        comparison.put("recallAtK", recall);
        comparison.put("hnswLatencyMs", hnswTimeMs);
        comparison.put("exactLatencyMs", exactTimeMs);
        comparison.put("speedupFactor", exactTimeMs > 0 ? (exactTimeMs / (hnswTimeMs > 0 ? hnswTimeMs : 0.001)) : 1.0);
        comparison.put("totalIndexedDocuments", hnswIndex.size());
        comparison.put("hnswResults", buildResultItems(hnswResults));
        comparison.put("exactResults", buildResultItems(exactResults));

        return comparison;
    }

    private void validateRequest(SearchRequest request) {
        if (request == null) {
            throw new IllegalArgumentException("SearchRequest cannot be null");
        }
        if (request.getQuery() == null || request.getQuery().trim().isEmpty()) {
            throw new IllegalArgumentException("Search query must not be blank");
        }
        if (request.getTopK() <= 0) {
            throw new IllegalArgumentException("topK must be greater than 0");
        }
        if (!indexManagementService.isReady()) {
            throw new IllegalStateException("Search index is still initializing. Please try again shortly.");
        }
    }

    private List<SearchResultResponse> buildResultItems(List<HnswSearchResult> results) {
        if (results == null || results.isEmpty()) {
            return Collections.emptyList();
        }

        List<Long> ids = results.stream().map(HnswSearchResult::getNodeId).toList();
        Map<Long, Document> docMap = documentRepository.findAllById(ids).stream()
                .collect(Collectors.toMap(Document::getId, d -> d));

        List<SearchResultResponse> responses = new ArrayList<>(results.size());
        for (HnswSearchResult res : results) {
            Document doc = docMap.get(res.getNodeId());
            if (doc != null) {
                String excerpt = createExcerpt(doc.getContent(), 200);
                responses.add(new SearchResultResponse(
                        doc.getId(),
                        doc.getTitle(),
                        excerpt,
                        res.getSimilarity(),
                        res.getDistance()
                ));
            }
        }
        return responses;
    }

    private String createExcerpt(String content, int maxLen) {
        if (content == null || content.length() <= maxLen) {
            return content;
        }
        return content.substring(0, maxLen).trim() + "...";
    }
}
