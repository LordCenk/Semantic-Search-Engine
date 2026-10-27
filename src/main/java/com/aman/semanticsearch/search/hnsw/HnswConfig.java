package com.aman.semanticsearch.search.hnsw;

/**
 * Configuration parameters for the Hierarchical Navigable Small World (HNSW) graph index.
 *
 * Parameters and their trade-offs:
 * - M (Graph Connectivity):
 *     The maximum number of bidirectional connections per node on layers > 0.
 *     Layer 0 uses M_max0 = 2 * M to ensure denser local connectivity.
 *     Higher M improves graph recall and resilience to local minima, but increases memory usage
 *     and index construction time. Typical range: 8 to 64.
 *
 * - efConstruction (Build Exploration Factor):
 *     The size of the dynamic candidate list evaluated during index construction.
 *     Higher efConstruction leads to better graph connectivity and higher search recall,
 *     at the cost of slower index insertion. Typical range: 32 to 200.
 *
 * - efSearch (Query Exploration Factor):
 *     The size of the dynamic candidate list evaluated during nearest-neighbor queries.
 *     Higher efSearch increases search accuracy (recall@K) while trading off query latency.
 *     Typical range: 16 to 128.
 *
 * - mL (Layer Normalization Factor):
 *     Scaling parameter for probabilistic layer assignment: mL = 1.0 / ln(M).
 *     Ensures exponential decay in layer sizes analogous to a skip-list.
 */
public class HnswConfig {

    private final int m;
    private final int mMax0;
    private final int efConstruction;
    private final int efSearch;
    private final double mL;

    public HnswConfig() {
        this(16, 64, 32);
    }

    public HnswConfig(int m, int efConstruction, int efSearch) {
        if (m < 2) {
            throw new IllegalArgumentException("M must be at least 2");
        }
        if (efConstruction < m) {
            throw new IllegalArgumentException("efConstruction must be >= M");
        }
        if (efSearch < 1) {
            throw new IllegalArgumentException("efSearch must be at least 1");
        }

        this.m = m;
        this.mMax0 = 2 * m;
        this.efConstruction = efConstruction;
        this.efSearch = efSearch;
        this.mL = 1.0 / Math.log(m);
    }

    public int getM() {
        return m;
    }

    public int getMMax0() {
        return mMax0;
    }

    public int getEfConstruction() {
        return efConstruction;
    }

    public int getEfSearch() {
        return efSearch;
    }

    public double getML() {
        return mL;
    }

    @Override
    public String toString() {
        return "HnswConfig{" +
                "M=" + m +
                ", M_max0=" + mMax0 +
                ", efConstruction=" + efConstruction +
                ", efSearch=" + efSearch +
                ", mL=" + String.format("%.4f", mL) +
                '}';
    }
}
