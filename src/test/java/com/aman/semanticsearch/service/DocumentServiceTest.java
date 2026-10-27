package com.aman.semanticsearch.service;

import com.aman.semanticsearch.algorithm.ExactSearchService;
import com.aman.semanticsearch.dto.DocumentRequest;
import com.aman.semanticsearch.dto.DocumentResponse;
import com.aman.semanticsearch.exception.DocumentNotFoundException;
import com.aman.semanticsearch.model.Document;
import com.aman.semanticsearch.repository.DocumentRepository;
import com.aman.semanticsearch.search.hnsw.HnswIndex;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DocumentServiceTest {

    @Mock
    private DocumentRepository documentRepository;

    @Mock
    private EmbeddingService embeddingService;

    @Mock
    private HnswIndex hnswIndex;

    @Mock
    private ExactSearchService exactSearchService;

    private DocumentService documentService;

    @BeforeEach
    void setUp() {
        documentService = new DocumentService(
                documentRepository,
                embeddingService,
                hnswIndex,
                exactSearchService
        );
    }

    @Test
    @DisplayName("Creating document should generate embedding, persist to DB, and insert into HNSW")
    void testCreateDocument() {
        DocumentRequest req = new DocumentRequest("Spring Boot Intro", "Overview of Spring Boot features");
        float[] mockVec = new float[]{0.1f, 0.2f, 0.3f};

        when(embeddingService.generateEmbedding(anyString())).thenReturn(mockVec);

        Document savedDoc = new Document(req.getTitle(), req.getContent(), mockVec);
        savedDoc.setId(42L);
        when(documentRepository.save(any(Document.class))).thenReturn(savedDoc);

        DocumentResponse response = documentService.createDocument(req);

        assertNotNull(response);
        assertEquals(42L, response.getId());
        assertEquals("Spring Boot Intro", response.getTitle());
        assertEquals(3, response.getEmbeddingDimensions());

        verify(hnswIndex).insert(42L, mockVec);
        verify(exactSearchService).indexVector(42L, mockVec);
    }

    @Test
    @DisplayName("Getting non-existent document should throw DocumentNotFoundException")
    void testGetDocumentNotFound() {
        when(documentRepository.findById(999L)).thenReturn(Optional.empty());

        assertThrows(DocumentNotFoundException.class, () -> documentService.getDocument(999L));
    }

    @Test
    @DisplayName("Updating document should update text, re-generate embedding, and re-index")
    void testUpdateDocument() {
        float[] oldVec = new float[]{0.1f, 0.1f};
        Document existing = new Document("Old Title", "Old Content", oldVec);
        existing.setId(10L);

        when(documentRepository.findById(10L)).thenReturn(Optional.of(existing));

        float[] newVec = new float[]{0.9f, 0.9f};
        when(embeddingService.generateEmbedding(anyString())).thenReturn(newVec);

        Document updatedDoc = new Document("New Title", "New Content", newVec);
        updatedDoc.setId(10L);
        when(documentRepository.save(existing)).thenReturn(updatedDoc);

        DocumentRequest updateReq = new DocumentRequest("New Title", "New Content");
        DocumentResponse response = documentService.updateDocument(10L, updateReq);

        assertEquals("New Title", response.getTitle());
        verify(hnswIndex).insert(10L, newVec);
        verify(exactSearchService).indexVector(10L, newVec);
    }

    @Test
    @DisplayName("Deleting document should remove from repository and both search indexes")
    void testDeleteDocument() {
        when(documentRepository.existsById(5L)).thenReturn(true);

        documentService.deleteDocument(5L);

        verify(documentRepository).deleteById(5L);
        verify(hnswIndex).delete(5L);
        verify(exactSearchService).removeVector(5L);
    }
}
