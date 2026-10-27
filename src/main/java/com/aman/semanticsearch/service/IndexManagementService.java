package com.aman.semanticsearch.service;

import com.aman.semanticsearch.algorithm.ExactSearchService;
import com.aman.semanticsearch.algorithm.VectorSimilarity;
import com.aman.semanticsearch.model.Document;
import com.aman.semanticsearch.repository.DocumentRepository;
import com.aman.semanticsearch.search.hnsw.HnswIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Manages the lifecycle, loading, and rebuilding of the in-memory HNSW index
 * from MySQL persisted embeddings.
 */
@Service
public class IndexManagementService {

    private static final Logger log = LoggerFactory.getLogger(IndexManagementService.class);

    private final DocumentRepository documentRepository;
    private final HnswIndex hnswIndex;
    private final ExactSearchService exactSearchService;
    private final AtomicBoolean ready = new AtomicBoolean(false);

    public IndexManagementService(DocumentRepository documentRepository,
                                  HnswIndex hnswIndex,
                                  ExactSearchService exactSearchService) {
        this.documentRepository = documentRepository;
        this.hnswIndex = hnswIndex;
        this.exactSearchService = exactSearchService;
    }

    /**
     * Initializes the in-memory HNSW graph and Exact vector store upon application startup.
     * The index is made available for searches ONLY after full initialization completes successfully.
     */
    @EventListener(ApplicationReadyEvent.class)
    public synchronized void initializeIndex() {
        log.info("Starting HNSW index initialization from database...");
        try {
            ready.set(false);
            hnswIndex.clear();
            exactSearchService.clear();

            List<Document> documents = documentRepository.findAllWithEmbedding();
            log.info("Loaded {} documents with embeddings from database", documents.size());

            int indexedCount = 0;
            Integer commonDimension = null;

            for (Document doc : documents) {
                float[] emb = doc.getEmbedding();
                if (emb == null || emb.length == 0) {
                    continue;
                }

                if (commonDimension == null) {
                    commonDimension = emb.length;
                } else if (emb.length != commonDimension) {
                    throw new IllegalStateException(String.format(
                            "Inconsistent embedding dimensions in DB for document %d: expected %d, got %d",
                            doc.getId(), commonDimension, emb.length));
                }

                VectorSimilarity.validateVector(emb, commonDimension);
                hnswIndex.insert(doc.getId(), emb);
                exactSearchService.indexVector(doc.getId(), emb);
                indexedCount++;
            }

            ready.set(true);
            log.info("HNSW index successfully initialized with {} documents (Dimension: {}, MaxLayer: {})",
                    indexedCount, hnswIndex.getDimension(), hnswIndex.getMaxLayer());

        } catch (Exception e) {
            ready.set(false);
            log.error("Failed to initialize HNSW index: {}", e.getMessage(), e);
            throw new IllegalStateException("HNSW index initialization failed", e);
        }
    }

    /**
     * Rebuilds the HNSW graph index and Exact vector store from all persisted documents.
     */
    public synchronized void rebuildIndex() {
        initializeIndex();
    }

    /**
     * Checks if the index is ready to serve search queries.
     */
    public boolean isReady() {
        return ready.get();
    }
}
