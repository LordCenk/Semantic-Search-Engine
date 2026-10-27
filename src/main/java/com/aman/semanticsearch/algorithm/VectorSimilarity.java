package com.aman.semanticsearch.algorithm;

/**
 * High-performance vector arithmetic and similarity metrics for high-dimensional embeddings.
 *
 * Mathematical Foundations:
 * 1. Dot Product: u · v = Σ (u_i * v_i)
 * 2. L2 Norm (Euclidean Magnitude): ||u|| = sqrt(Σ u_i^2)
 * 3. Cosine Similarity: cos(θ) = (u · v) / (||u|| * ||v||)
 *    - Range: [-1.0, 1.0]
 *    - 1.0  => Vectors point in identical direction.
 *    - 0.0  => Vectors are orthogonal (no correlation).
 *    - -1.0 => Vectors point in opposite directions.
 * 4. Cosine Distance: d(u, v) = 1.0 - CosineSimilarity(u, v)
 *    - Range: [0.0, 2.0]
 *    - 0.0  => Identical vectors.
 *    - 1.0  => Orthogonal vectors.
 *    - 2.0  => Opposite vectors.
 *    - Monotonically decreases as similarity increases, making it ideal for nearest-neighbor min-heaps in HNSW.
 */
public final class VectorSimilarity {

    private VectorSimilarity() {
        // Utility class
    }

    /**
     * Calculates the Cosine Similarity between two float vectors.
     *
     * @param a first vector
     * @param b second vector
     * @return cosine similarity in range [-1.0, 1.0], or 0.0 if either vector has zero magnitude
     * @throws IllegalArgumentException if vectors are null, have mismatched dimensions, or contain NaN/Inf
     */
    public static double cosineSimilarity(float[] a, float[] b) {
        validateVectors(a, b);

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < a.length; i++) {
            float valA = a[i];
            float valB = b[i];

            if (Float.isNaN(valA) || Float.isInfinite(valA) || Float.isNaN(valB) || Float.isInfinite(valB)) {
                throw new IllegalArgumentException("Invalid vector value: NaN or Infinite encountered at index " + i);
            }

            dotProduct += (double) valA * valB;
            normA += (double) valA * valA;
            normB += (double) valB * valB;
        }

        // Handle zero vector edge-cases gracefully
        if (normA <= 0.0 || normB <= 0.0) {
            return 0.0;
        }

        double similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));

        // Clamp against floating-point inaccuracies slightly outside [-1.0, 1.0]
        if (similarity > 1.0) {
            return 1.0;
        } else if (similarity < -1.0) {
            return -1.0;
        }
        return similarity;
    }

    /**
     * Calculates the Cosine Distance between two float vectors.
     * Defined as: 1.0 - cosineSimilarity(a, b)
     *
     * @param a first vector
     * @param b second vector
     * @return cosine distance in range [0.0, 2.0]
     */
    public static double cosineDistance(float[] a, float[] b) {
        return 1.0 - cosineSimilarity(a, b);
    }

    /**
     * Calculates the Dot Product between two vectors.
     */
    public static double dotProduct(float[] a, float[] b) {
        validateVectors(a, b);
        double dot = 0.0;
        for (int i = 0; i < a.length; i++) {
            float valA = a[i];
            float valB = b[i];
            if (Float.isNaN(valA) || Float.isInfinite(valA) || Float.isNaN(valB) || Float.isInfinite(valB)) {
                throw new IllegalArgumentException("Invalid vector value: NaN or Infinite encountered at index " + i);
            }
            dot += (double) valA * valB;
        }
        return dot;
    }

    /**
     * Calculates the Euclidean (L2) distance between two vectors.
     */
    public static double euclideanDistance(float[] a, float[] b) {
        validateVectors(a, b);
        double sumSq = 0.0;
        for (int i = 0; i < a.length; i++) {
            float valA = a[i];
            float valB = b[i];
            if (Float.isNaN(valA) || Float.isInfinite(valA) || Float.isNaN(valB) || Float.isInfinite(valB)) {
                throw new IllegalArgumentException("Invalid vector value: NaN or Infinite encountered at index " + i);
            }
            double diff = (double) valA - valB;
            sumSq += diff * diff;
        }
        return Math.sqrt(sumSq);
    }

    /**
     * Validates that both vectors are non-null and have matching dimensions.
     */
    public static void validateVectors(float[] a, float[] b) {
        if (a == null) {
            throw new IllegalArgumentException("Vector a cannot be null");
        }
        if (b == null) {
            throw new IllegalArgumentException("Vector b cannot be null");
        }
        if (a.length == 0) {
            throw new IllegalArgumentException("Vector a cannot be empty");
        }
        if (a.length != b.length) {
            throw new IllegalArgumentException(String.format(
                    "Vector dimension mismatch: vector a has %d dimensions, but vector b has %d dimensions",
                    a.length, b.length));
        }
    }

    /**
     * Validates that a single vector is non-null, non-empty, and all elements are finite.
     */
    public static void validateVector(float[] vector, int expectedDimension) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector cannot be null");
        }
        if (vector.length == 0) {
            throw new IllegalArgumentException("Vector cannot be empty");
        }
        if (expectedDimension > 0 && vector.length != expectedDimension) {
            throw new IllegalArgumentException(String.format(
                    "Invalid vector dimension: expected %d, got %d", expectedDimension, vector.length));
        }
        for (int i = 0; i < vector.length; i++) {
            float val = vector[i];
            if (Float.isNaN(val) || Float.isInfinite(val)) {
                throw new IllegalArgumentException("Invalid vector value: NaN or Infinite encountered at index " + i);
            }
        }
    }
}
