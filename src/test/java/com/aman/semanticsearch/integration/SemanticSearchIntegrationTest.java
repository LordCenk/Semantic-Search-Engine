package com.aman.semanticsearch.integration;

import com.aman.semanticsearch.dto.DocumentRequest;
import com.aman.semanticsearch.dto.SearchRequest;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import static org.hamcrest.Matchers.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class SemanticSearchIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    @DisplayName("End-to-end integration: add documents, search with HNSW and Exact, compare, and delete")
    void testEndToEndSemanticSearchFlow() throws Exception {
        // 1. Add first document
        DocumentRequest doc1 = new DocumentRequest(
                "Introduction to Spring Boot",
                "Spring Boot makes it easy to create stand-alone, production-grade Spring based Applications."
        );
        MvcResult doc1Result = mockMvc.perform(post("/api/documents")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(doc1)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").isNumber())
                .andExpect(jsonPath("$.title").value("Introduction to Spring Boot"))
                .andExpect(jsonPath("$.embeddingDimensions").value(32))
                .andReturn();

        Number doc1Id = objectMapper.readTree(doc1Result.getResponse().getContentAsString()).get("id").numberValue();

        // 2. Add second document
        DocumentRequest doc2 = new DocumentRequest(
                "HNSW Algorithm Explained",
                "Hierarchical Navigable Small World graphs provide efficient approximate nearest neighbor search."
        );
        mockMvc.perform(post("/api/documents")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(doc2)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.title").value("HNSW Algorithm Explained"));

        // 3. Add third document
        DocumentRequest doc3 = new DocumentRequest(
                "Cooking Italian Pasta",
                "Boil water with salt, cook pasta until al dente, and toss with olive oil and parmesan cheese."
        );
        mockMvc.perform(post("/api/documents")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(doc3)))
                .andExpect(status().isCreated());

        // 4. Retrieve document by ID
        mockMvc.perform(get("/api/documents/" + doc1Id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.title").value("Introduction to Spring Boot"));

        // 5. List documents with pagination
        mockMvc.perform(get("/api/documents?page=0&size=10"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.content", hasSize(greaterThanOrEqualTo(3))));

        // 6. Perform Semantic Search with HNSW
        SearchRequest hnswSearch = new SearchRequest("How to build applications with Spring Boot?", 2, "HNSW");
        mockMvc.perform(post("/api/search")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(hnswSearch)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.searchMethod").value("HNSW"))
                .andExpect(jsonPath("$.results", hasSize(lessThanOrEqualTo(2))))
                .andExpect(jsonPath("$.results[0].title").exists());

        // 7. Perform Semantic Search with Exact brute-force baseline
        SearchRequest exactSearch = new SearchRequest("Hierarchical nearest neighbor search", 2, "EXACT");
        mockMvc.perform(post("/api/search")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(exactSearch)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.searchMethod").value("EXACT"))
                .andExpect(jsonPath("$.results[0].title").value("HNSW Algorithm Explained"));

        // 8. Run Compare endpoint (HNSW vs Exact + Recall@K)
        SearchRequest compareReq = new SearchRequest("Spring and HNSW architecture", 2);
        mockMvc.perform(post("/api/search/compare")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(compareReq)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.recallAtK").isNumber())
                .andExpect(jsonPath("$.hnswLatencyMs").isNumber())
                .andExpect(jsonPath("$.exactLatencyMs").isNumber());

        // 9. Trigger index rebuild
        mockMvc.perform(post("/api/search/rebuild"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.message").exists());

        // 10. Delete document and verify it is removed
        mockMvc.perform(delete("/api/documents/" + doc1Id))
                .andExpect(status().isNoContent());

        mockMvc.perform(get("/api/documents/" + doc1Id))
                .andExpect(status().isNotFound());
    }

    @Autowired
    private com.aman.semanticsearch.search.hnsw.HnswIndex hnswIndex;

    @Autowired
    private com.aman.semanticsearch.service.IndexManagementService indexManagementService;

    @Test
    @DisplayName("Verify index survives application restart: persisted embeddings are reloaded and HNSW graph is reconstructed")
    void testIndexSurvivesApplicationRestart() throws Exception {
        // 1. Ingest a distinct document that is stored in the database
        DocumentRequest req = new DocumentRequest(
                "Microservice Resiliency with Circuit Breakers",
                "Fault tolerance patterns like Circuit Breaker and Rate Limiter prevent cascading failures."
        );
        MvcResult result = mockMvc.perform(post("/api/documents")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isCreated())
                .andReturn();

        Number savedId = objectMapper.readTree(result.getResponse().getContentAsString()).get("id").numberValue();
        org.junit.jupiter.api.Assertions.assertTrue(hnswIndex.contains(savedId.longValue()));

        // 2. Simulate Application Crash / Restart:
        // Wiping in-memory state completely
        hnswIndex.clear();
        org.junit.jupiter.api.Assertions.assertEquals(0, hnswIndex.size(), "In-memory HNSW index should be completely empty");

        // 3. Trigger startup initialization (what Spring Boot executes on ApplicationReadyEvent)
        indexManagementService.initializeIndex();

        // 4. Verify in-memory HNSW graph has been completely reconstructed from database records
        org.junit.jupiter.api.Assertions.assertTrue(hnswIndex.size() >= 1, "HNSW index must have recovered documents from MySQL/H2");
        org.junit.jupiter.api.Assertions.assertTrue(hnswIndex.contains(savedId.longValue()), "Recovered graph must contain the persisted document ID");

        // 5. Verify semantic search works seamlessly against the reconstructed graph
        SearchRequest searchReq = new SearchRequest("Circuit breaker fault tolerance in microservices", 2, "HNSW");
        mockMvc.perform(post("/api/search")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(searchReq)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.results[0].documentId").value(savedId.longValue()))
                .andExpect(jsonPath("$.results[0].title").value("Microservice Resiliency with Circuit Breakers"));
    }
}
