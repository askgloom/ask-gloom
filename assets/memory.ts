import { EventEmitter } from 'events';
import { VectorStore, type Vector } from './vector';
import { generateHash, calculateSimilarity } from './utils';

export interface MemoryConfig {
    capacity: number;              // Maximum number of memories
    decayRate: number;            // Rate at which memory importance decays
    retrievalThreshold: number;   // Minimum similarity for retrieval
    embedDimension?: number;      // Dimension of memory embeddings
    consolidationInterval?: number; // Time between memory consolidations
}

export interface MemoryEntry {
    id: string;
    content: string | object;
    type?: string;
    timestamp: number;
    importance: number;
    metadata?: Record<string, any>;
    embedding?: Vector;
    lastAccessed?: number;
    accessCount?: number;
}

export interface RetrievalParams {
    limit?: number;
    minRelevance?: number;
    type?: string;
    timeRange?: {
        start: number;
        end: number;
    };
}

export class Memory extends EventEmitter {
    private store: VectorStore;
    private config: Required<MemoryConfig>;
    private entries: Map<string, MemoryEntry>;
    private consolidationTimer?: NodeJS.Timer;

    constructor(config: MemoryConfig) {
        super();
        this.config = {
            embedDimension: 384,
            consolidationInterval: 3600000, // 1 hour
            ...config
        };
        
        this.store = new VectorStore(this.config.embedDimension);
        this.entries = new Map();
        
        this.initializeConsolidation();
    }

    async store(entry: Omit<MemoryEntry, 'id'>): Promise<string> {
        try {
            const id = generateHash(entry.content);
            const embedding = await this.generateEmbedding(entry.content);
            
            const memoryEntry: MemoryEntry = {
                id,
                embedding,
                accessCount: 0,
                lastAccessed: Date.now(),
                ...entry
            };

            // Check capacity and potentially remove lowest importance memory
            if (this.entries.size >= this.config.capacity) {
                await this.removeLowestImportanceMemory();
            }

            // Store memory
            this.entries.set(id, memoryEntry);
            await this.store.add({ id, vector: embedding, metadata: memoryEntry });

            this.emit('memoryStored', memoryEntry);
            return id;
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }

    async retrieve(query: string, params: RetrievalParams = {}): Promise<MemoryEntry[]> {
        try {
            const queryEmbedding = await this.generateEmbedding(query);
            const {
                limit = 10,
                minRelevance = this.config.retrievalThreshold,
                type,
                timeRange
            } = params;

            // Search vector store
            const results = await this.store.search(queryEmbedding, {
                limit: limit * 2, // Get extra results for filtering
                minSimilarity: minRelevance
            });

            // Filter and process results
            const filtered = results.filter(result => {
                const memory = this.entries.get(result.id);
                if (!memory) return false;

                // Apply filters
                if (type && memory.type !== type) return false;
                if (timeRange) {
                    const { start, end } = timeRange;
                    if (memory.timestamp < start || memory.timestamp > end) return false;
                }

                return true;
            }).slice(0, limit);

            // Update access metrics
            filtered.forEach(result => {
                const memory = this.entries.get(result.id);
                if (memory) {
                    memory.lastAccessed = Date.now();
                    memory.accessCount = (memory.accessCount || 0) + 1;
                }
            });

            this.emit('memoriesRetrieved', filtered);
            return filtered.map(r => this.entries.get(r.id)!);
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }

    async update(id: string, updates: Partial<MemoryEntry>): Promise<void> {
        try {
            const entry = this.entries.get(id);
            if (!entry) throw new Error(`Memory with id ${id} not found`);

            const updatedEntry = { ...entry, ...updates };
            
            // If content changed, update embedding
            if ('content' in updates) {
                updatedEntry.embedding = await this.generateEmbedding(updates.content!);
                await this.store.update(id, updatedEntry.embedding);
            }

            this.entries.set(id, updatedEntry);
            this.emit('memoryUpdated', updatedEntry);
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }

    async remove(id: string): Promise<void> {
        try {
            const entry = this.entries.get(id);
            if (!entry) throw new Error(`Memory with id ${id} not found`);

            this.entries.delete(id);
            await this.store.remove(id);
            
            this.emit('memoryRemoved', entry);
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }

    async consolidate(): Promise<void> {
        try {
            const memories = Array.from(this.entries.values());
            const consolidated = await this.consolidateMemories(memories);
            
            // Clear and rebuild store
            await this.store.clear();
            this.entries.clear();

            // Store consolidated memories
            for (const memory of consolidated) {
                await this.store(memory);
            }

            this.emit('memoriesConsolidated', consolidated);
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }

    async getStats(): Promise<any> {
        return {
            totalMemories: this.entries.size,
            averageImportance: this.calculateAverageImportance(),
            oldestMemory: this.getOldestMemory(),
            newestMemory: this.getNewestMemory(),
            memoryTypes: this.getMemoryTypeCounts()
        };
    }

    // Private helper methods
    private async generateEmbedding(content: string | object): Promise<Vector> {
        // Implementation depends on your embedding strategy
        return [];  // Placeholder
    }

    private async removeLowestImportanceMemory(): Promise<void> {
        let lowest = { id: '', importance: Infinity };
        
        for (const [id, memory] of this.entries) {
            if (memory.importance < lowest.importance) {
                lowest = { id, importance: memory.importance };
            }
        }

        if (lowest.id) {
            await this.remove(lowest.id);
        }
    }

    private async consolidateMemories(memories: MemoryEntry[]): Promise<MemoryEntry[]> {
        // Implement your memory consolidation strategy
        return memories;  // Placeholder
    }

    private initializeConsolidation(): void {
        if (this.config.consolidationInterval > 0) {
            this.consolidationTimer = setInterval(
                () => this.consolidate(),
                this.config.consolidationInterval
            );
        }
    }

    private calculateAverageImportance(): number {
        const sum = Array.from(this.entries.values())
            .reduce((acc, memory) => acc + memory.importance, 0);
        return sum / this.entries.size || 0;
    }

    private getOldestMemory(): MemoryEntry | null {
        return Array.from(this.entries.values())
            .reduce((oldest, current) => 
                current.timestamp < (oldest?.timestamp ?? Infinity) ? current : oldest, null);
    }

    private getNewestMemory(): MemoryEntry | null {
        return Array.from(this.entries.values())
            .reduce((newest, current) => 
                current.timestamp > (newest?.timestamp ?? -Infinity) ? current : newest, null);
    }

    private getMemoryTypeCounts(): Record<string, number> {
        return Array.from(this.entries.values())
            .reduce((counts, memory) => {
                const type = memory.type || 'unknown';
                counts[type] = (counts[type] || 0) + 1;
                return counts;
            }, {} as Record<string, number>);
    }
}