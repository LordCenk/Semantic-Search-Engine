package com.aman.semanticsearch.service;

import com.aman.semanticsearch.algorithm.VectorSimilarity;
import com.aman.semanticsearch.exception.EmbeddingException;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Random;

/**
 * Deterministic pseudo-embedding provider for offline development, repeatable automated tests,
 * and environments where external paid API credentials are not provided.
 *
 * NOTE: This implementation generates deterministic mathematical vectors derived from text hashes.
 * While it guarantees that identical texts produce identical vectors and overlapping terms share
 * directional components, it does NOT represent true deep-learning semantic understanding.
 */
public class DeterministicFakeEmbeddingService implements EmbeddingService {

    private final int dimensions;
    private final String modelName;

    public DeterministicFakeEmbeddingService(int dimensions, String modelName) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive");
        }
        this.dimensions = dimensions;
        this.modelName = modelName != null ? modelName : "fake-deterministic-model";
    }

    @Override
    public float[] generateEmbedding(String text) {
        if (text == null || text.trim().isEmpty()) {
            throw new EmbeddingException("Cannot generate embedding for null or blank text");
        }

        float[] vector = new float[dimensions];
        String cleaned = text.toLowerCase().trim();
        String[] tokens = cleaned.split("\\s+");

        // Hash-based bag-of-words projection:
        // Each word contributes deterministically to specific dimensions using SHA-256
        for (String token : tokens) {
            long seed = hashTokenToLong(token);
            Random rng = new Random(seed);

            // Project token onto dimensions with random Gaussian weights
            for (int d = 0; d < dimensions; d++) {
                vector[d] += (float) rng.nextGaussian();
            }
        }

        // Add overall document hash to differentiate sentences with similar words
        long fullHash = hashTokenToLong(cleaned);
        Random docRng = new Random(fullHash);
        for (int d = 0; d < dimensions; d++) {
            vector[d] += (float) (docRng.nextGaussian() * 0.5);
        }

        // Normalize vector to unit length (L2 norm = 1.0)
        double normSq = 0.0;
        for (float v : vector) {
            normSq += (double) v * v;
        }

        double norm = Math.sqrt(normSq);
        if (norm > 0.0) {
            for (int i = 0; i < dimensions; i++) {
                vector[i] = (float) (vector[i] / norm);
            }
        }

        // Validate vector
        VectorSimilarity.validateVector(vector, dimensions);
        return vector;
    }

    @Override
    public int getDimensions() {
        return dimensions;
    }

    @Override
    public String getModelName() {
        return modelName;
    }

    private long hashTokenToLong(String token) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(token.getBytes(StandardCharsets.UTF_8));
            long result = 0;
            for (int i = 0; i < 8; i++) {
                result = (result << 8) | (hash[i] & 0xFF);
            }
            return result;
        } catch (NoSuchAlgorithmException e) {
            return token.hashCode();
        }
    }
}
