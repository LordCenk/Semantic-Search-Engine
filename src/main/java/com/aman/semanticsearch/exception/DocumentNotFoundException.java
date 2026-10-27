package com.aman.semanticsearch.exception;

public class DocumentNotFoundException extends RuntimeException {

    public DocumentNotFoundException(Long id) {
        super("Document not found with ID: " + id);
    }

    public DocumentNotFoundException(String message) {
        super(message);
    }
}
