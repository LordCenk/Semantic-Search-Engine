package com.aman.semanticsearch.repository;

import com.aman.semanticsearch.model.Document;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Spring Data JPA repository for Document persistence.
 * Note: Responsible strictly for data access and persistence, NOT for HNSW index logic.
 */
@Repository
public interface DocumentRepository extends JpaRepository<Document, Long> {

    /**
     * Efficiently load all documents that have non-null embeddings for building the in-memory HNSW index.
     */
    @Query("SELECT d FROM Document d WHERE d.embedding IS NOT NULL")
    List<Document> findAllWithEmbedding();
}
