package com.aman.semanticsearch.controller;

import com.aman.semanticsearch.dto.SearchRequest;
import com.aman.semanticsearch.dto.SearchResponse;
import com.aman.semanticsearch.service.IndexManagementService;
import com.aman.semanticsearch.service.SemanticSearchService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/search")
@Tag(name = "Semantic Search", description = "Endpoints for approximate HNSW and exact vector search queries")
public class SearchController {

    private final SemanticSearchService searchService;
    private final IndexManagementService indexManagementService;

    public SearchController(SemanticSearchService searchService,
                            IndexManagementService indexManagementService) {
        this.searchService = searchService;
        this.indexManagementService = indexManagementService;
    }

    @PostMapping
    @Operation(summary = "Search documents", description = "Executes natural-language semantic vector search")
    public ResponseEntity<SearchResponse> search(@Valid @RequestBody SearchRequest request) {
        return ResponseEntity.ok(searchService.search(request));
    }

    @PostMapping("/compare")
    @Operation(summary = "Compare HNSW and Exact Search", description = "Runs both searches to benchmark latency and measure Recall@K")
    public ResponseEntity<Map<String, Object>> compare(@Valid @RequestBody SearchRequest request) {
        return ResponseEntity.ok(searchService.compare(request));
    }

    @PostMapping("/rebuild")
    @Operation(summary = "Rebuild Search Index", description = "Rebuilds the in-memory HNSW graph index from persisted database embeddings")
    public ResponseEntity<Map<String, String>> rebuildIndex() {
        indexManagementService.rebuildIndex();
        return ResponseEntity.ok(Map.of("message", "HNSW search index rebuild completed successfully."));
    }
}
