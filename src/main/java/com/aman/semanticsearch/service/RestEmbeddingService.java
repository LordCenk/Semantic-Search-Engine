package com.aman.semanticsearch.service;

import com.aman.semanticsearch.algorithm.VectorSimilarity;
import com.aman.semanticsearch.exception.EmbeddingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.*;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

/**
 * Real REST API client for external embedding providers (e.g., OpenAI, Ollama, Hugging Face TEI).
 *
 * Supported formats:
 * - OpenAI style: POST /v1/embeddings with {"input": text, "model": model}
 *   Response: {"data": [{"embedding": [...]}]}
 * - Ollama style: POST /api/embeddings with {"prompt": text, "model": model}
 *   Response: {"embedding": [...]}
 */
public class RestEmbeddingService implements EmbeddingService {

    private static final Logger log = LoggerFactory.getLogger(RestEmbeddingService.class);

    private final String apiUrl;
    private final String apiKey;
    private final String modelName;
    private final int expectedDimensions;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public RestEmbeddingService(String apiUrl, String apiKey, String modelName, int expectedDimensions, int timeoutMs) {
        if (apiUrl == null || apiUrl.trim().isEmpty()) {
            throw new IllegalArgumentException("API URL cannot be null or empty");
        }
        this.apiUrl = apiUrl;
        this.apiKey = apiKey != null ? apiKey : "";
        this.modelName = modelName != null ? modelName : "text-embedding-3-small";
        this.expectedDimensions = expectedDimensions;

        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(timeoutMs);
        factory.setReadTimeout(timeoutMs);
        this.restTemplate = new RestTemplate(factory);
    }

    @Override
    public float[] generateEmbedding(String text) {
        if (text == null || text.trim().isEmpty()) {
            throw new EmbeddingException("Cannot generate embedding for blank text");
        }

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            if (!apiKey.isBlank()) {
                // Attach Authorization header if API key is provided
                headers.setBearerAuth(apiKey);
            }

            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("model", modelName);

            // Adapt payload based on endpoint URL
            if (apiUrl.contains("ollama")) {
                requestBody.put("prompt", text);
            } else {
                requestBody.put("input", text);
            }

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);

            log.debug("Sending embedding request to: {}", apiUrl);
            ResponseEntity<String> response = restTemplate.postForEntity(apiUrl, requestEntity, String.class);

            if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
                throw new EmbeddingException("Provider returned non-2xx status: " + response.getStatusCode());
            }

            return parseEmbeddingFromJson(response.getBody());

        } catch (RestClientResponseException e) {
            log.error("Embedding provider returned error HTTP {}: {}", e.getStatusCode(), e.getStatusText());
            throw new EmbeddingException("Embedding provider error (" + e.getStatusCode() + "): " + e.getResponseBodyAsString(), e);
        } catch (Exception e) {
            log.error("Failed to connect to embedding service at {}: {}", apiUrl, e.getMessage());
            throw new EmbeddingException("Failed to generate embedding: " + e.getMessage(), e);
        }
    }

    private float[] parseEmbeddingFromJson(String json) {
        try {
            JsonNode root = objectMapper.readTree(json);
            JsonNode vectorNode = null;

            // Check OpenAI style: data[0].embedding
            if (root.has("data") && root.get("data").isArray() && !root.get("data").isEmpty()) {
                JsonNode first = root.get("data").get(0);
                if (first.has("embedding")) {
                    vectorNode = first.get("embedding");
                }
            } else if (root.has("embedding")) {
                // Check Ollama style: embedding
                vectorNode = root.get("embedding");
            }

            if (vectorNode == null || !vectorNode.isArray()) {
                throw new EmbeddingException("Unable to find numerical embedding array in API response");
            }

            int dim = vectorNode.size();
            if (dim == 0) {
                throw new EmbeddingException("Received empty embedding array from provider");
            }

            float[] vector = new float[dim];
            for (int i = 0; i < dim; i++) {
                vector[i] = (float) vectorNode.get(i).asDouble();
            }

            // Validate dimensions against configured expectation (if specified)
            VectorSimilarity.validateVector(vector, expectedDimensions > 0 ? expectedDimensions : dim);
            return vector;

        } catch (EmbeddingException e) {
            throw e;
        } catch (Exception e) {
            throw new EmbeddingException("Error parsing embedding JSON response: " + e.getMessage(), e);
        }
    }

    @Override
    public int getDimensions() {
        return expectedDimensions;
    }

    @Override
    public String getModelName() {
        return modelName;
    }
}
