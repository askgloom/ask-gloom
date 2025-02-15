// Core exports
export { Agent } from './agent';
export { Memory } from './memory';
export { VectorStore } from './vector';

// Type exports
export type {
    AgentConfig,
    AgentOptions,
    AgentResponse,
    ProcessOptions,
    Message,
    Conversation
} from './types';

export type {
    MemoryConfig,
    MemoryEntry,
    RetrievalParams,
    Vector,
    VectorSearchParams,
    VectorSearchResult
} from './memory';

// Utility exports
export {
    generateHash,
    calculateSimilarity,
    generateEmbedding,
    validateConfig
} from './utils';

// Constants
export const VERSION = '0.1.0';

// Error types
export class GloomError extends Error {
    constructor(message: string, public code: string) {
        super(message);
        this.name = 'GloomError';
    }
}

export class AgentError extends GloomError {
    constructor(message: string) {
        super(message, 'AGENT_ERROR');
        this.name = 'AgentError';
    }
}

export class MemoryError extends GloomError {
    constructor(message: string) {
        super(message, 'MEMORY_ERROR');
        this.name = 'MemoryError';
    }
}

// Default configurations
export const DEFAULT_AGENT_CONFIG: AgentConfig = {
    name: 'default',
    memory: {
        capacity: 1000,
        decayRate: 0.1,
        retrievalThreshold: 0.5
    },
    maxTokens: 2048,
    temperature: 0.7
};

export const DEFAULT_MEMORY_CONFIG: MemoryConfig = {
    capacity: 1000,
    decayRate: 0.1,
    retrievalThreshold: 0.5,
    embedDimension: 384,
    consolidationInterval: 3600000 // 1 hour
};

// Event names
export const EVENTS = {
    AGENT: {
        INITIALIZED: 'agent:initialized',
        PROCESSING: 'agent:processing',
        RESPONSE: 'agent:response',
        ERROR: 'agent:error'
    },
    MEMORY: {
        STORED: 'memory:stored',
        RETRIEVED: 'memory:retrieved',
        CONSOLIDATED: 'memory:consolidated',
        ERROR: 'memory:error'
    }
} as const;

// Middleware types
export type Middleware<T = any> = (
    context: T,
    next: () => Promise<void>
) => Promise<void>;

// Plugin interface
export interface Plugin {
    name: string;
    version: string;
    install: (agent: Agent) => void | Promise<void>;
}

// Re-export commonly used utilities
export { z } from 'zod';