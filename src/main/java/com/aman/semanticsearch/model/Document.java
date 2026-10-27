package com.aman.semanticsearch.model;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.persistence.*;
import java.time.Instant;
import java.util.Arrays;

/**
 * Entity representing a text document and its high-dimensional semantic embedding vector.
 */
@Entity
@Table(name = "documents")
public class Document {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 500)
    private String title;

    @Lob
    @Column(nullable = false, columnDefinition = "LONGTEXT")
    private String content;

    /**
     * High-dimensional embedding vector safely serialized as JSON array of numbers.
     * Prevents dangerous native Java serialization and provides standard database compatibility.
     */
    @Convert(converter = FloatArrayJsonConverter.class)
    @Lob
    @Column(columnDefinition = "LONGTEXT")
    private float[] embedding;

    @Column(nullable = false, updatable = false)
    private Instant createdAt;

    @Column(nullable = false)
    private Instant updatedAt;

    public Document() {
    }

    public Document(String title, String content, float[] embedding) {
        this.title = title;
        this.content = content;
        this.embedding = embedding;
    }

    @PrePersist
    protected void onCreate() {
        Instant now = Instant.now();
        this.createdAt = now;
        this.updatedAt = now;
    }

    @PreUpdate
    protected void onUpdate() {
        this.updatedAt = Instant.now();
    }

    // Getters and Setters

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public float[] getEmbedding() {
        return embedding;
    }

    public void setEmbedding(float[] embedding) {
        this.embedding = embedding;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(Instant updatedAt) {
        this.updatedAt = updatedAt;
    }

    @Override
    public String toString() {
        return "Document{" +
                "id=" + id +
                ", title='" + title + '\'' +
                ", embeddingDim=" + (embedding != null ? embedding.length : 0) +
                ", createdAt=" + createdAt +
                ", updatedAt=" + updatedAt +
                '}';
    }

    /**
     * Safe JSON converter for float array embeddings.
     */
    @Converter
    public static class FloatArrayJsonConverter implements AttributeConverter<float[], String> {

        private static final ObjectMapper MAPPER = new ObjectMapper();

        @Override
        public String convertToDatabaseColumn(float[] attribute) {
            if (attribute == null) {
                return null;
            }
            try {
                return MAPPER.writeValueAsString(attribute);
            } catch (JsonProcessingException e) {
                throw new IllegalStateException("Failed to serialize float[] embedding to JSON string", e);
            }
        }

        @Override
        public float[] convertToEntityAttribute(String dbData) {
            if (dbData == null || dbData.trim().isEmpty()) {
                return null;
            }
            try {
                return MAPPER.readValue(dbData, float[].class);
            } catch (JsonProcessingException e) {
                throw new IllegalStateException("Failed to deserialize JSON string to float[] embedding", e);
            }
        }
    }
}
