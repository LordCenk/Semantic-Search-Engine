package com.aman.semanticsearch.service;

/**
 * Strategy interface for text embedding generation.
 * Decouples the semantic search engine from specific embedding providers
 * (e.g., OpenAI, Ollama, Hugging Face, or local test providers).
 */
public interface EmbeddingService {

    /**
     * Converts raw text into a normalized numerical embedding vector.
     *
     * @param text document or search query text
     * @return non-empty float vector of fixed dimensions
     * @throws com.aman.semanticsearch.exception.EmbeddingException if generation fails or input is invalid
     */
    float[] generateEmbedding(String text);

    /**
     * Returns the fixed dimensionality of the embedding vectors produced by this service.
     */
    int getDimensions();

    /**
     * Returns the name of the underlying embedding model or provider.
     */
    String getModelName();
}
