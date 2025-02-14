import { Agent, Memory, type AgentConfig } from '@gloom/core';
import { MemoryUtils } from '@gloom/toolkit';

// Basic configuration for our agent
const config: AgentConfig = {
    name: 'SimpleAgent',
    memory: {
        capacity: 1000,           // Maximum memory entries
        decayRate: 0.1,          // Rate at which memories fade
        retrievalThreshold: 0.5   // Minimum relevance for memory recall
    }
};

// Create a new agent instance
const agent = new Agent(config);

// Example memory entries
const memories = [
    {
        content: "The sky is blue",
        timestamp: Date.now(),
        importance: 0.7
    },
    {
        content: "Water is wet",
        timestamp: Date.now(),
        importance: 0.8
    }
];

// Store memories
memories.forEach(memory => {
    agent.memory.store(memory);
});

// Example query function
async function queryAgent(question: string): Promise<string> {
    try {
        // Retrieve relevant memories
        const relevantMemories = await agent.memory.retrieve(question);
        
        // Process the query with context
        const response = await agent.process({
            input: question,
            context: relevantMemories
        });

        return response;
    } catch (error) {
        console.error('Error processing query:', error);
        return 'I encountered an error processing your question.';
    }
}

// Example usage
async function main() {
    // Query the agent
    const response = await queryAgent("What do you know about the sky?");
    console.log('Agent Response:', response);

    // Get memory statistics
    const stats = await agent.memory.getStats();
    console.log('Memory Stats:', stats);
}

main().catch(console.error);

// Export for use in other examples
export { agent, queryAgent };