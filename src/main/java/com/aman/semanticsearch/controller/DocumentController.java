package com.aman.semanticsearch.controller;

import com.aman.semanticsearch.dto.DocumentRequest;
import com.aman.semanticsearch.dto.DocumentResponse;
import com.aman.semanticsearch.service.DocumentService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/documents")
@Tag(name = "Document Management", description = "Endpoints for creating, retrieving, updating, and deleting documents")
public class DocumentController {

    private final DocumentService documentService;

    public DocumentController(DocumentService documentService) {
        this.documentService = documentService;
    }

    @PostMapping
    @Operation(summary = "Add a document", description = "Persists document text, generates embedding, and updates HNSW index")
    public ResponseEntity<DocumentResponse> addDocument(@Valid @RequestBody DocumentRequest request) {
        DocumentResponse response = documentService.createDocument(request);
        return new ResponseEntity<>(response, HttpStatus.CREATED);
    }

    @GetMapping("/{id}")
    @Operation(summary = "Get a document by ID")
    public ResponseEntity<DocumentResponse> getDocument(@PathVariable Long id) {
        return ResponseEntity.ok(documentService.getDocument(id));
    }

    @GetMapping
    @Operation(summary = "List documents with pagination")
    public ResponseEntity<Page<DocumentResponse>> listDocuments(@PageableDefault(size = 20) Pageable pageable) {
        return ResponseEntity.ok(documentService.getDocuments(pageable));
    }

    @PutMapping("/{id}")
    @Operation(summary = "Update a document", description = "Updates text, regenerates embedding, and updates HNSW index")
    public ResponseEntity<DocumentResponse> updateDocument(@PathVariable Long id, @Valid @RequestBody DocumentRequest request) {
        return ResponseEntity.ok(documentService.updateDocument(id, request));
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "Delete a document", description = "Deletes document from database and removes from HNSW graph")
    public ResponseEntity<Void> deleteDocument(@PathVariable Long id) {
        documentService.deleteDocument(id);
        return ResponseEntity.noContent().build();
    }
}
