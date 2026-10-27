package com.aman.semanticsearch.benchmark;

import com.aman.semanticsearch.algorithm.ExactSearchService;
import com.aman.semanticsearch.search.hnsw.HnswConfig;
import com.aman.semanticsearch.search.hnsw.HnswIndex;
import com.aman.semanticsearch.search.hnsw.HnswSearchResult;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Standalone, reproducible benchmark runner comparing:
 * 1. Brute-Force Exact Search with Full Sort (O(N log N))
 * 2. Exact Search with Top-K Bounded Max-Heap (O(N log K))
 * 3. HNSW Approximate Nearest Neighbor Graph Navigation (O(log N))
 *
 * Measures:
 * - Index construction time
 * - Search latency percentiles (Average, Median/p50, p90, p95, p99)
 * - Queries Per Second (QPS)
 * - Recall@K against exact nearest-neighbor ground truth
 * - Memory usage before and after indexing
 */
public class HnswBenchmarkRunner {

    private static final int DIMENSIONS = 64;
    private static final int TOP_K = 10;
    private static final int QUERY_COUNT = 100;
    private static final int WARMUP_COUNT = 200;

    public static void main(String[] args) throws Exception {
        System.out.println("==================================================================");
        System.out.println("   HNSW vs Exact Vector Search Algorithmic Benchmark Suite");
        System.out.println("==================================================================");

        int[] datasetSizes = {1_000, 10_000, 50_000};
        if (args.length > 0 && !args[0].isEmpty()) {
            String[] custom = args[0].split(",");
            datasetSizes = new int[custom.length];
            for (int i = 0; i < custom.length; i++) {
                datasetSizes[i] = Integer.parseInt(custom[i].trim());
            }
        }

        List<Map<String, Object>> allResults = new ArrayList<>();

        for (int size : datasetSizes) {
            Map<String, Object> run = benchmarkDatasetSize(size);
            allResults.add(run);
        }

        saveResultsToJson(allResults, "benchmark_results.json");
        printSummaryTable(allResults);
    }

    private static Map<String, Object> benchmarkDatasetSize(int numDocs) {
        System.out.printf("\n>>> Running Benchmark for N = %,d documents (Dimension = %d, Top-K = %d) <<<\n",
                numDocs, DIMENSIONS, TOP_K);

        Random rng = new Random(1337);

        // 1. Generate synthetic dataset
        System.out.println("Generating normalized synthetic vectors...");
        List<float[]> dataset = new ArrayList<>(numDocs);
        for (int i = 0; i < numDocs; i++) {
            dataset.add(generateRandomUnitVector(DIMENSIONS, rng));
        }

        // Generate query vectors
        List<float[]> queries = new ArrayList<>(QUERY_COUNT);
        for (int i = 0; i < QUERY_COUNT; i++) {
            queries.add(generateRandomUnitVector(DIMENSIONS, rng));
        }

        // 2. Measure Memory Before
        System.gc();
        long memBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        // 3. Build Exact Store
        ExactSearchService exactService = new ExactSearchService();
        for (long i = 0; i < numDocs; i++) {
            exactService.indexVector(i, dataset.get((int) i));
        }

        // 4. Build HNSW Index
        HnswConfig config = new HnswConfig(16, 64, 32);
        HnswIndex hnswIndex = new HnswIndex(config, new Random(42));

        long buildStart = System.currentTimeMillis();
        for (long i = 0; i < numDocs; i++) {
            hnswIndex.insert(i, dataset.get((int) i));
            if (numDocs >= 10000 && (i + 1) % 2500 == 0) {
                System.out.printf("  Indexed %,d / %,d documents (%.1f%%)...\n",
                        i + 1, numDocs, ((i + 1) * 100.0) / numDocs);
            }
        }
        long buildTimeMs = System.currentTimeMillis() - buildStart;
        System.out.printf("HNSW Index Build Time: %,d ms (%.2f docs/sec)\n",
                buildTimeMs, (numDocs * 1000.0) / Math.max(buildTimeMs, 1));

        // Measure Memory After
        System.gc();
        long memAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        double memUsedMb = Math.max(0, memAfter - memBefore) / (1024.0 * 1024.0);

        // 5. Warm up JVM Hotspot JIT compiler
        System.out.println("Warming up JVM HotSpot JIT compiler...");
        for (int w = 0; w < WARMUP_COUNT; w++) {
            float[] q = queries.get(w % queries.size());
            hnswIndex.search(q, TOP_K);
            exactService.search(q, TOP_K);
        }

        // 6. Benchmark HNSW Search
        System.out.println("Executing HNSW queries...");
        List<Double> hnswLatencies = new ArrayList<>(QUERY_COUNT);
        List<List<HnswSearchResult>> hnswResultSets = new ArrayList<>(QUERY_COUNT);
        for (float[] q : queries) {
            long t0 = System.nanoTime();
            List<HnswSearchResult> res = hnswIndex.search(q, TOP_K);
            long t1 = System.nanoTime();
            hnswLatencies.add((t1 - t0) / 1_000_000.0);
            hnswResultSets.add(res);
        }

        // 7. Benchmark Exact Search (Max-Heap)
        System.out.println("Executing Exact Max-Heap queries...");
        List<Double> exactHeapLatencies = new ArrayList<>(QUERY_COUNT);
        List<List<HnswSearchResult>> exactResultSets = new ArrayList<>(QUERY_COUNT);
        for (float[] q : queries) {
            long t0 = System.nanoTime();
            List<HnswSearchResult> res = exactService.search(q, TOP_K);
            long t1 = System.nanoTime();
            exactHeapLatencies.add((t1 - t0) / 1_000_000.0);
            exactResultSets.add(res);
        }

        // 8. Benchmark Exact Search with Full Sort (for small-to-medium datasets)
        List<Double> exactSortLatencies = new ArrayList<>();
        if (numDocs <= 10_000) {
            System.out.println("Executing Exact Full-Sort queries...");
            for (float[] q : queries) {
                long t0 = System.nanoTime();
                exactService.searchWithFullSort(q, TOP_K);
                long t1 = System.nanoTime();
                exactSortLatencies.add((t1 - t0) / 1_000_000.0);
            }
        }

        // 9. Calculate Recall@K
        double totalRecall = 0.0;
        for (int i = 0; i < QUERY_COUNT; i++) {
            totalRecall += ExactSearchService.calculateRecall(hnswResultSets.get(i), exactResultSets.get(i));
        }
        double avgRecall = totalRecall / QUERY_COUNT;

        LatencyStats hnswStats = computeStats(hnswLatencies);
        LatencyStats exactHeapStats = computeStats(exactHeapLatencies);
        LatencyStats exactSortStats = !exactSortLatencies.isEmpty() ? computeStats(exactSortLatencies) : null;

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("numDocs", numDocs);
        result.put("dimension", DIMENSIONS);
        result.put("topK", TOP_K);
        result.put("buildTimeMs", buildTimeMs);
        result.put("approxMemoryMb", memUsedMb);
        result.put("hnswMaxLayer", hnswIndex.getMaxLayer());
        result.put("recallAtK", avgRecall);
        result.put("hnswAvgLatencyMs", hnswStats.avg);
        result.put("hnswP50Ms", hnswStats.p50);
        result.put("hnswP95Ms", hnswStats.p95);
        result.put("hnswP99Ms", hnswStats.p99);
        result.put("hnswQPS", hnswStats.qps);
        result.put("exactHeapAvgLatencyMs", exactHeapStats.avg);
        result.put("exactHeapP95Ms", exactHeapStats.p95);
        result.put("exactHeapQPS", exactHeapStats.qps);
        if (exactSortStats != null) {
            result.put("exactSortAvgLatencyMs", exactSortStats.avg);
            result.put("exactSortP95Ms", exactSortStats.p95);
        }
        result.put("speedupVsExactHeap", exactHeapStats.avg / Math.max(hnswStats.avg, 0.0001));

        return result;
    }

    private static LatencyStats computeStats(List<Double> latencies) {
        List<Double> sorted = new ArrayList<>(latencies);
        Collections.sort(sorted);
        double sum = 0.0;
        for (double d : sorted) sum += d;
        double avg = sum / sorted.size();
        double p50 = sorted.get((int) (sorted.size() * 0.50));
        double p90 = sorted.get((int) (sorted.size() * 0.90));
        double p95 = sorted.get((int) (sorted.size() * 0.95));
        double p99 = sorted.get((int) (sorted.size() * 0.99));
        double qps = 1000.0 / Math.max(avg, 0.0001);
        return new LatencyStats(avg, p50, p90, p95, p99, qps);
    }

    private static float[] generateRandomUnitVector(int dim, Random rng) {
        float[] v = new float[dim];
        double sumSq = 0.0;
        for (int i = 0; i < dim; i++) {
            v[i] = (float) rng.nextGaussian();
            sumSq += v[i] * v[i];
        }
        double norm = Math.sqrt(sumSq);
        for (int i = 0; i < dim; i++) {
            v[i] = (float) (v[i] / norm);
        }
        return v;
    }

    private static void printSummaryTable(List<Map<String, Object>> results) {
        System.out.println("\n==========================================================================================================");
        System.out.println("                                         BENCHMARK SUMMARY RESULTS");
        System.out.println("==========================================================================================================");
        System.out.printf("%-10s | %-12s | %-10s | %-12s | %-12s | %-12s | %-12s | %-10s\n",
                "Docs (N)", "Build Time", "Recall@10", "HNSW Avg (ms)", "HNSW p95 (ms)", "Exact Avg (ms)", "HNSW QPS", "Speedup");
        System.out.println("----------------------------------------------------------------------------------------------------------");
        for (Map<String, Object> r : results) {
            System.out.printf("%-10s | %-10d ms | %-10.2f%% | %-12.4f | %-12.4f | %-12.4f | %-12.1f | %-8.1fx\n",
                    String.format("%,d", (int) r.get("numDocs")),
                    (long) r.get("buildTimeMs"),
                    ((double) r.get("recallAtK")) * 100.0,
                    (double) r.get("hnswAvgLatencyMs"),
                    (double) r.get("hnswP95Ms"),
                    (double) r.get("exactHeapAvgLatencyMs"),
                    (double) r.get("hnswQPS"),
                    (double) r.get("speedupVsExactHeap"));
        }
        System.out.println("==========================================================================================================");
    }

    private static void saveResultsToJson(List<Map<String, Object>> results, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("[\n");
            for (int i = 0; i < results.size(); i++) {
                Map<String, Object> r = results.get(i);
                writer.write("  {\n");
                int count = 0;
                for (Map.Entry<String, Object> e : r.entrySet()) {
                    count++;
                    String val = (e.getValue() instanceof Number) ? e.getValue().toString() : "\"" + e.getValue() + "\"";
                    writer.write("    \"" + e.getKey() + "\": " + val + (count < r.size() ? ",\n" : "\n"));
                }
                writer.write("  }" + (i < results.size() - 1 ? ",\n" : "\n"));
            }
            writer.write("]\n");
            System.out.println("\nSaved benchmark results to " + filename);
        } catch (IOException e) {
            System.err.println("Failed to write benchmark results file: " + e.getMessage());
        }
    }

    private static class LatencyStats {
        final double avg;
        final double p50;
        final double p90;
        final double p95;
        final double p99;
        final double qps;

        LatencyStats(double avg, double p50, double p90, double p95, double p99, double qps) {
            this.avg = avg;
            this.p50 = p50;
            this.p90 = p90;
            this.p95 = p95;
            this.p99 = p99;
            this.qps = qps;
        }
    }
}
