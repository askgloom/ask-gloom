import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import { CLI } from '../src/cli';
import { Agent } from '@gloom/core';
import fs from 'fs/promises';
import path from 'path';

// Mock dependencies
jest.mock('@gloom/core');
jest.mock('fs/promises');
jest.mock('../src/commands/init');
jest.mock('../src/commands/run');

describe('CLI', () => {
    let cli: CLI;
    const mockConsole = {
        log: jest.fn(),
        error: jest.fn()
    };

    beforeEach(() => {
        // Clear all mocks
        jest.clearAllMocks();
        
        // Initialize CLI with mocked console
        cli = new CLI({
            console: mockConsole
        });
    });

    afterEach(() => {
        jest.resetAllMocks();
    });

    describe('initialization', () => {
        it('should create a new CLI instance', () => {
            expect(cli).toBeInstanceOf(CLI);
        });

        it('should initialize with default options', () => {
            const defaultCli = new CLI();
            expect(defaultCli).toHaveProperty('version');
            expect(defaultCli).toHaveProperty('commands');
        });
    });

    describe('command execution', () => {
        it('should execute init command successfully', async () => {
            const args = ['init', '--name', 'test-agent'];
            await cli.execute(args);
            
            expect(mockConsole.log).toHaveBeenCalledWith(
                expect.stringContaining('Initialized')
            );
        });

        it('should execute run command successfully', async () => {
            const args = ['run', '--input', 'test query'];
            await cli.execute(args);
            
            expect(mockConsole.log).toHaveBeenCalledWith(
                expect.stringContaining('Response')
            );
        });

        it('should handle unknown commands', async () => {
            const args = ['unknown-command'];
            await cli.execute(args);
            
            expect(mockConsole.error).toHaveBeenCalledWith(
                expect.stringContaining('Unknown command')
            );
        });
    });

    describe('configuration management', () => {
        const mockConfig = {
            name: 'test-agent',
            memory: {
                capacity: 1000,
                decayRate: 0.1
            }
        };

        beforeEach(() => {
            (fs.readFile as jest.Mock).mockResolvedValue(
                JSON.stringify(mockConfig)
            );
        });

        it('should load configuration file', async () => {
            const config = await cli.loadConfig();
            expect(config).toEqual(mockConfig);
        });

        it('should handle missing configuration file', async () => {
            (fs.readFile as jest.Mock).mockRejectedValue(new Error('ENOENT'));
            
            await expect(cli.loadConfig()).rejects.toThrow(
                'Configuration file not found'
            );
        });

        it('should save configuration file', async () => {
            await cli.saveConfig(mockConfig);
            
            expect(fs.writeFile).toHaveBeenCalledWith(
                expect.any(String),
                JSON.stringify(mockConfig, null, 2)
            );
        });
    });

    describe('agent interaction', () => {
        const mockAgent = {
            process: jest.fn().mockResolvedValue('test response'),
            memory: {
                store: jest.fn(),
                retrieve: jest.fn()
            }
        };

        beforeEach(() => {
            (Agent as jest.Mock).mockImplementation(() => mockAgent);
        });

        it('should create and initialize agent', async () => {
            await cli.initializeAgent({
                name: 'test-agent',
                memory: { capacity: 1000 }
            });

            expect(Agent).toHaveBeenCalled();
        });

        it('should process queries through agent', async () => {
            await cli.initializeAgent({ name: 'test-agent' });
            const response = await cli.processQuery('test query');
            
            expect(mockAgent.process).toHaveBeenCalledWith(
                expect.objectContaining({
                    input: 'test query'
                })
            );
            expect(response).toBe('test response');
        });

        it('should handle agent errors', async () => {
            mockAgent.process.mockRejectedValue(new Error('Agent error'));
            
            await cli.initializeAgent({ name: 'test-agent' });
            await expect(cli.processQuery('test query')).rejects.toThrow('Agent error');
        });
    });

    describe('error handling', () => {
        it('should handle invalid arguments', async () => {
            const args = ['run']; // Missing required --input
            await cli.execute(args);
            
            expect(mockConsole.error).toHaveBeenCalledWith(
                expect.stringContaining('Invalid arguments')
            );
        });

        it('should handle runtime errors', async () => {
            const error = new Error('Runtime error');
            (Agent as jest.Mock).mockImplementation(() => {
                throw error;
            });

            await cli.execute(['init', '--name', 'test-agent']);
            
            expect(mockConsole.error).toHaveBeenCalledWith(
                expect.stringContaining('Runtime error')
            );
        });
    });

    describe('utility functions', () => {
        it('should validate configuration', () => {
            const validConfig = {
                name: 'test-agent',
                memory: {
                    capacity: 1000,
                    decayRate: 0.1
                }
            };

            expect(() => cli.validateConfig(validConfig)).not.toThrow();
        });

        it('should format help message', () => {
            const help = cli.formatHelp();
            expect(help).toContain('Usage');
            expect(help).toContain('Commands');
            expect(help).toContain('Options');
        });

        it('should parse command line arguments', () => {
            const args = ['run', '--input', 'test query', '--model', 'gpt-4'];
            const parsed = cli.parseArgs(args);
            
            expect(parsed).toEqual({
                command: 'run',
                options: {
                    input: 'test query',
                    model: 'gpt-4'
                }
            });
        });
    });
});