import { z } from 'zod';
import type { MemoryEntry, AgentConfig, MemoryConfig } from '@gloom/core';

// Basic validation schemas
const vectorSchema = z.array(z.number())
    .refine(arr => arr.every(n => !isNaN(n)), {
        message: 'Vector must contain only valid numbers'
    });

// Memory validation schemas
const memoryMetadataSchema = z.record(z.unknown()).optional();

const memoryEntrySchema = z.object({
    id: z.string().optional(),
    content: z.union([z.string(), z.record(z.unknown())]),
    type: z.string().optional(),
    timestamp: z.number().int().positive(),
    importance: z.number().min(0).max(1),
    metadata: memoryMetadataSchema,
    embedding: vectorSchema.optional(),
    lastAccessed: z.number().int().positive().optional(),
    accessCount: z.number().int().min(0).optional()
});

// Configuration validation schemas
const memoryConfigSchema = z.object({
    capacity: z.number().int().positive(),
    decayRate: z.number().min(0).max(1),
    retrievalThreshold: z.number().min(0).max(1),
    embedDimension: z.number().int().positive().optional(),
    consolidationInterval: z.number().int().positive().optional()
});

const agentConfigSchema = z.object({
    name: z.string().min(1),
    memory: memoryConfigSchema,
    modelConfig: z.record(z.unknown()).optional(),
    maxTokens: z.number().int().positive().optional(),
    temperature: z.number().min(0).max(2).optional()
});

// Validation functions
export class Validation {
    /**
     * Validates a memory entry
     */
    static validateMemoryEntry(entry: unknown): MemoryEntry {
        try {
            return memoryEntrySchema.parse(entry);
        } catch (error) {
            throw this.formatValidationError('Invalid memory entry', error);
        }
    }

    /**
     * Validates memory configuration
     */
    static validateMemoryConfig(config: unknown): MemoryConfig {
        try {
            return memoryConfigSchema.parse(config);
        } catch (error) {
            throw this.formatValidationError('Invalid memory configuration', error);
        }
    }

    /**
     * Validates agent configuration
     */
    static validateAgentConfig(config: unknown): AgentConfig {
        try {
            return agentConfigSchema.parse(config);
        } catch (error) {
            throw this.formatValidationError('Invalid agent configuration', error);
        }
    }

    /**
     * Validates a vector
     */
    static validateVector(vector: unknown): number[] {
        try {
            return vectorSchema.parse(vector);
        } catch (error) {
            throw this.formatValidationError('Invalid vector', error);
        }
    }

    /**
     * Validates memory metadata
     */
    static validateMetadata(metadata: unknown): Record<string, unknown> | undefined {
        try {
            return memoryMetadataSchema.parse(metadata);
        } catch (error) {
            throw this.formatValidationError('Invalid metadata', error);
        }
    }

    /**
     * Validates memory importance value
     */
    static validateImportance(importance: unknown): number {
        try {
            return z.number().min(0).max(1).parse(importance);
        } catch (error) {
            throw this.formatValidationError('Invalid importance value', error);
        }
    }

    /**
     * Validates timestamp
     */
    static validateTimestamp(timestamp: unknown): number {
        try {
            return z.number().int().positive().parse(timestamp);
        } catch (error) {
            throw this.formatValidationError('Invalid timestamp', error);
        }
    }

    /**
     * Batch validation of memory entries
     */
    static validateMemoryBatch(entries: unknown[]): MemoryEntry[] {
        try {
            return z.array(memoryEntrySchema).parse(entries);
        } catch (error) {
            throw this.formatValidationError('Invalid memory batch', error);
        }
    }

    /**
     * Validates retrieval parameters
     */
    static validateRetrievalParams(params: unknown): {
        limit?: number;
        minRelevance?: number;
        type?: string;
        timeRange?: { start: number; end: number };
    } {
        const schema = z.object({
            limit: z.number().int().positive().optional(),
            minRelevance: z.number().min(0).max(1).optional(),
            type: z.string().optional(),
            timeRange: z.object({
                start: z.number().int().positive(),
                end: z.number().int().positive()
            }).optional()
        });

        try {
            return schema.parse(params);
        } catch (error) {
            throw this.formatValidationError('Invalid retrieval parameters', error);
        }
    }

    // Helper methods
    private static formatValidationError(message: string, error: unknown): Error {
        if (error instanceof z.ZodError) {
            const details = error.errors.map(err => 
                `${err.path.join('.')}: ${err.message}`
            ).join('; ');
            return new Error(`${message}: ${details}`);
        }
        return new Error(message);
    }

    /**
     * Type guard for memory entry
     */
    static isMemoryEntry(value: unknown): value is MemoryEntry {
        try {
            memoryEntrySchema.parse(value);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Type guard for memory config
     */
    static isMemoryConfig(value: unknown): value is MemoryConfig {
        try {
            memoryConfigSchema.parse(value);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Type guard for agent config
     */
    static isAgentConfig(value: unknown): value is AgentConfig {
        try {
            agentConfigSchema.parse(value);
            return true;
        } catch {
            return false;
        }
    }
}

export const validate = Validation;