package com.aman.semanticsearch.config;

import com.aman.semanticsearch.search.hnsw.HnswConfig;
import com.aman.semanticsearch.search.hnsw.HnswIndex;
import com.aman.semanticsearch.service.DeterministicFakeEmbeddingService;
import com.aman.semanticsearch.service.EmbeddingService;
import com.aman.semanticsearch.service.RestEmbeddingService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Spring configuration for Embedding services and HNSW Index beans.
 */
@Configuration
public class EmbeddingConfig {

    private static final Logger log = LoggerFactory.getLogger(EmbeddingConfig.class);

    @Bean
    public EmbeddingService embeddingService(SearchProperties properties) {
        SearchProperties.Embedding config = properties.getEmbedding();
        String provider = config.getProvider() != null ? config.getProvider().trim().toLowerCase() : "fake";

        if ("openai".equals(provider) || "ollama".equals(provider) || "rest".equals(provider)) {
            log.info("Configuring REST embedding provider: {} [model: {}, url: {}]",
                    provider, config.getModel(), config.getApiUrl());
            return new RestEmbeddingService(
                    config.getApiUrl(),
                    config.getApiKey(),
                    config.getModel(),
                    config.getDimensions(),
                    config.getTimeoutMs()
            );
        }

        log.info("Configuring deterministic fake embedding provider [dimensions: {}]", config.getDimensions());
        return new DeterministicFakeEmbeddingService(config.getDimensions(), "deterministic-fake-model");
    }

    @Bean
    public HnswConfig hnswConfig(SearchProperties properties) {
        SearchProperties.Hnsw hnswProps = properties.getHnsw();
        HnswConfig config = new HnswConfig(hnswProps.getM(), hnswProps.getEfConstruction(), hnswProps.getEfSearch());
        log.info("Initialized HNSW configuration: {}", config);
        return config;
    }

    @Bean
    public HnswIndex hnswIndex(HnswConfig hnswConfig) {
        return new HnswIndex(hnswConfig);
    }
}
