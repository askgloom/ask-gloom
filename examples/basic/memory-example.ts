import { Memory, type MemoryConfig, type MemoryEntry } from '@gloom/core';
import { MemoryUtils } from '@gloom/toolkit';

// Configure memory system
const config: MemoryConfig = {
    capacity: 1000,
    decayRate: 0.1,
    retrievalThreshold: 0.5,
    indexingStrategy: 'semantic'  // Use semantic indexing for better retrieval
};

// Initialize memory system
const memory = new Memory(config);

// Example memory entries with different types
const memories: MemoryEntry[] = [
    {
        content: "User asked about machine learning",
        type: 'interaction',
        timestamp: Date.now(),
        importance: 0.8,
        metadata: {
            topic: 'AI',
            sentiment: 'curious'
        }
    },
    {
        content: "System explained neural networks",
        type: 'knowledge',
        timestamp: Date.now(),
        importance: 0.9,
        metadata: {
            topic: 'AI',
            confidence: 0.95
        }
    }
];

// Store memories with importance scoring
async function storeMemories() {
    for (const entry of memories) {
        const importance = await MemoryUtils.calculateImportance(entry);
        await memory.store({
            ...entry,
            importance
        });
    }
}

// Retrieve memories based on query
async function searchMemories(query: string) {
    try {
        const results = await memory.retrieve(query, {
            limit: 5,
            minRelevance: 0.6
        });
        return results;
    } catch (error) {
        console.error('Error retrieving memories:', error);
        return [];
    }
}

// Memory maintenance and optimization
async function optimizeMemory() {
    await memory.consolidate();  // Merge similar memories
    await memory.prune();        // Remove low importance memories
    
    const stats = await memory.getStats();
    console.log('Memory Stats:', stats);
}

// Example usage
async function main() {
    await storeMemories();
    
    console.log('Searching for AI-related memories...');
    const results = await searchMemories('machine learning');
    console.log('Retrieved Memories:', results);
    
    await optimizeMemory();
}

main().catch(console.error);

export { memory, storeMemories, searchMemories };