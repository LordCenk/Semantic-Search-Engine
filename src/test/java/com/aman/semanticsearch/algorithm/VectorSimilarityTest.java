package com.aman.semanticsearch.algorithm;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class VectorSimilarityTest {

    private static final double EPSILON = 1e-6;

    @Test
    @DisplayName("Identical vectors should have cosine similarity 1.0 and distance 0.0")
    void testIdenticalVectors() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {1.0f, 2.0f, 3.0f};

        double similarity = VectorSimilarity.cosineSimilarity(a, b);
        double distance = VectorSimilarity.cosineDistance(a, b);

        assertEquals(1.0, similarity, EPSILON);
        assertEquals(0.0, distance, EPSILON);
    }

    @Test
    @DisplayName("Collinear scaled vectors should have cosine similarity 1.0")
    void testCollinearVectors() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {2.0f, 4.0f, 6.0f};

        double similarity = VectorSimilarity.cosineSimilarity(a, b);
        assertEquals(1.0, similarity, EPSILON);
    }

    @Test
    @DisplayName("Orthogonal vectors should have cosine similarity 0.0 and distance 1.0")
    void testOrthogonalVectors() {
        float[] a = {1.0f, 0.0f, 0.0f};
        float[] b = {0.0f, 1.0f, 0.0f};

        double similarity = VectorSimilarity.cosineSimilarity(a, b);
        double distance = VectorSimilarity.cosineDistance(a, b);

        assertEquals(0.0, similarity, EPSILON);
        assertEquals(1.0, distance, EPSILON);
    }

    @Test
    @DisplayName("Opposite vectors should have cosine similarity -1.0 and distance 2.0")
    void testOppositeVectors() {
        float[] a = {2.0f, 5.0f};
        float[] b = {-2.0f, -5.0f};

        double similarity = VectorSimilarity.cosineSimilarity(a, b);
        double distance = VectorSimilarity.cosineDistance(a, b);

        assertEquals(-1.0, similarity, EPSILON);
        assertEquals(2.0, distance, EPSILON);
    }

    @Test
    @DisplayName("Zero vector edge case should safely return similarity 0.0 and distance 1.0")
    void testZeroVector() {
        float[] zero = {0.0f, 0.0f, 0.0f};
        float[] nonZero = {1.0f, 2.0f, 3.0f};

        double similarity = VectorSimilarity.cosineSimilarity(zero, nonZero);
        double distance = VectorSimilarity.cosineDistance(zero, nonZero);

        assertEquals(0.0, similarity, EPSILON);
        assertEquals(1.0, distance, EPSILON);

        // Both zero vectors
        assertEquals(0.0, VectorSimilarity.cosineSimilarity(zero, zero), EPSILON);
        assertEquals(1.0, VectorSimilarity.cosineDistance(zero, zero), EPSILON);
    }

    @Test
    @DisplayName("Dimension mismatch should throw IllegalArgumentException")
    void testDimensionMismatch() {
        float[] a = {1.0f, 2.0f};
        float[] b = {1.0f, 2.0f, 3.0f};

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                VectorSimilarity.cosineSimilarity(a, b));
        assertTrue(ex.getMessage().contains("Vector dimension mismatch"));
    }

    @Test
    @DisplayName("Null or empty vectors should throw IllegalArgumentException")
    void testNullOrEmptyVectors() {
        float[] valid = {1.0f, 2.0f};

        assertThrows(IllegalArgumentException.class, () -> VectorSimilarity.cosineSimilarity(null, valid));
        assertThrows(IllegalArgumentException.class, () -> VectorSimilarity.cosineSimilarity(valid, null));
        assertThrows(IllegalArgumentException.class, () -> VectorSimilarity.cosineSimilarity(new float[0], valid));
    }

    @Test
    @DisplayName("Vectors with NaN or Infinity should throw IllegalArgumentException")
    void testInvalidFloatValues() {
        float[] nanVector = {1.0f, Float.NaN, 3.0f};
        float[] infVector = {1.0f, Float.POSITIVE_INFINITY, 3.0f};
        float[] valid = {1.0f, 2.0f, 3.0f};

        assertThrows(IllegalArgumentException.class, () -> VectorSimilarity.cosineSimilarity(nanVector, valid));
        assertThrows(IllegalArgumentException.class, () -> VectorSimilarity.cosineSimilarity(infVector, valid));
    }
}
