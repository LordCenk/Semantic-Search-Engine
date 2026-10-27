package com.aman.semanticsearch.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public class DocumentRequest {

    @NotBlank(message = "Title must not be blank")
    @Size(max = 500, message = "Title must be at most 500 characters")
    private String title;

    @NotBlank(message = "Content must not be blank")
    private String content;

    public DocumentRequest() {
    }

    public DocumentRequest(String title, String content) {
        this.title = title;
        this.content = content;
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
}
