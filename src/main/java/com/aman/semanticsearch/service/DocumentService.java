package com.aman.semanticsearch.service;

import com.aman.semanticsearch.algorithm.ExactSearchService;
import com.aman.semanticsearch.dto.DocumentRequest;
import com.aman.semanticsearch.dto.DocumentResponse;
import com.aman.semanticsearch.exception.DocumentNotFoundException;
import com.aman.semanticsearch.model.Document;
import com.aman.semanticsearch.repository.DocumentRepository;
import com.aman.semanticsearch.search.hnsw.HnswIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Service managing document lifecycle, persistence, and synchronization with the HNSW graph index.
 */
@Service
public class DocumentService {

    private static final Logger log = LoggerFactory.getLogger(DocumentService.class);

    private final DocumentRepository documentRepository;
    private final EmbeddingService embeddingService;
    private final HnswIndex hnswIndex;
    private final ExactSearchService exactSearchService;

    public DocumentService(DocumentRepository documentRepository,
                           EmbeddingService embeddingService,
                           HnswIndex hnswIndex,
                           ExactSearchService exactSearchService) {
        this.documentRepository = documentRepository;
        this.embeddingService = embeddingService;
        this.hnswIndex = hnswIndex;
        this.exactSearchService = exactSearchService;
    }

    /**
     * Creates and indexes a new document.
     */
    @Transactional
    public DocumentResponse createDocument(DocumentRequest request) {
        log.info("Creating document with title: {}", request.getTitle());

        // Generate embedding vector for the combined text representation
        String textToEmbed = buildSearchableText(request.getTitle(), request.getContent());
        float[] embedding = embeddingService.generateEmbedding(textToEmbed);

        // Persist document and embedding in database
        Document doc = new Document(request.getTitle(), request.getContent(), embedding);
        Document saved = documentRepository.save(doc);

        // Update in-memory search indices
        hnswIndex.insert(saved.getId(), embedding);
        exactSearchService.indexVector(saved.getId(), embedding);

        log.info("Document persisted and indexed with ID: {}", saved.getId());
        return mapToResponse(saved);
    }

    /**
     * Retrieves a document by ID.
     */
    @Transactional(readOnly = true)
    public DocumentResponse getDocument(Long id) {
        Document doc = documentRepository.findById(id)
                .orElseThrow(() -> new DocumentNotFoundException(id));
        return mapToResponse(doc);
    }

    /**
     * Lists documents with pagination.
     */
    @Transactional(readOnly = true)
    public Page<DocumentResponse> getDocuments(Pageable pageable) {
        return documentRepository.findAll(pageable).map(this::mapToResponse);
    }

    /**
     * Updates an existing document and refreshes its embedding and graph index connections.
     */
    @Transactional
    public DocumentResponse updateDocument(Long id, DocumentRequest request) {
        log.info("Updating document ID: {}", id);
        Document existing = documentRepository.findById(id)
                .orElseThrow(() -> new DocumentNotFoundException(id));

        existing.setTitle(request.getTitle());
        existing.setContent(request.getContent());

        // Regenerate embedding vector since content changed
        String textToEmbed = buildSearchableText(request.getTitle(), request.getContent());
        float[] newEmbedding = embeddingService.generateEmbedding(textToEmbed);
        existing.setEmbedding(newEmbedding);

        Document updated = documentRepository.save(existing);

        // Re-index into HNSW and Exact store
        hnswIndex.insert(updated.getId(), newEmbedding);
        exactSearchService.indexVector(updated.getId(), newEmbedding);

        return mapToResponse(updated);
    }

    /**
     * Deletes a document by ID and removes it from search indices.
     */
    @Transactional
    public void deleteDocument(Long id) {
        log.info("Deleting document ID: {}", id);
        if (!documentRepository.existsById(id)) {
            throw new DocumentNotFoundException(id);
        }

        documentRepository.deleteById(id);
        hnswIndex.delete(id);
        exactSearchService.removeVector(id);
    }

    private String buildSearchableText(String title, String content) {
        return title + "\n\n" + content;
    }

    private DocumentResponse mapToResponse(Document doc) {
        int dim = doc.getEmbedding() != null ? doc.getEmbedding().length : 0;
        return new DocumentResponse(
                doc.getId(),
                doc.getTitle(),
                doc.getContent(),
                dim,
                doc.getCreatedAt(),
                doc.getUpdatedAt()
        );
    }
}
