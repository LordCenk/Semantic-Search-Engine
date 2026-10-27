package com.aman.semanticsearch.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "search")
public class SearchProperties {

    private Hnsw hnsw = new Hnsw();
    private Embedding embedding = new Embedding();

    public Hnsw getHnsw() {
        return hnsw;
    }

    public void setHnsw(Hnsw hnsw) {
        this.hnsw = hnsw;
    }

    public Embedding getEmbedding() {
        return embedding;
    }

    public void setEmbedding(Embedding embedding) {
        this.embedding = embedding;
    }

    public static class Hnsw {
        private int m = 16;
        private int efConstruction = 64;
        private int efSearch = 32;

        public int getM() {
            return m;
        }

        public void setM(int m) {
            this.m = m;
        }

        public int getEfConstruction() {
            return efConstruction;
        }

        public void setEfConstruction(int efConstruction) {
            this.efConstruction = efConstruction;
        }

        public int getEfSearch() {
            return efSearch;
        }

        public void setEfSearch(int efSearch) {
            this.efSearch = efSearch;
        }
    }

    public static class Embedding {
        private String provider = "fake";
        private String apiKey = "";
        private String apiUrl = "https://api.openai.com/v1/embeddings";
        private String model = "text-embedding-3-small";
        private int dimensions = 128;
        private int timeoutMs = 10000;

        public String getProvider() {
            return provider;
        }

        public void setProvider(String provider) {
            this.provider = provider;
        }

        public String getApiKey() {
            return apiKey;
        }

        public void setApiKey(String apiKey) {
            this.apiKey = apiKey;
        }

        public String getApiUrl() {
            return apiUrl;
        }

        public void setApiUrl(String apiUrl) {
            this.apiUrl = apiUrl;
        }

        public String getModel() {
            return model;
        }

        public void setModel(String model) {
            this.model = model;
        }

        public int getDimensions() {
            return dimensions;
        }

        public void setDimensions(int dimensions) {
            this.dimensions = dimensions;
        }

        public int getTimeoutMs() {
            return timeoutMs;
        }

        public void setTimeoutMs(int timeoutMs) {
            this.timeoutMs = timeoutMs;
        }
    }
}
