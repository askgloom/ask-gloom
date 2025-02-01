import asyncio
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import traceback
import signal
import resource
from enum import Enum
import psutil
import json

class ExecutorState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class ExecutionResult:
    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    runtime: float = 0.0
    resources: Dict[str, float] = None
    metadata: Dict = None

class TaskExecutor:
    def __init__(self, max_workers: int = 5, resource_limits: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        self.resource_limits = resource_limits or {
            'cpu_percent': 80.0,
            'memory_percent': 70.0,
            'timeout': 3600  # 1 hour default timeout
        }
        
        # Execution state
        self.state = ExecutorState.IDLE
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.results: Dict[str, ExecutionResult] = {}
        
        # Resource monitoring
        self.process = psutil.Process()
        self.resource_usage: List[Dict] = []
        
        # Initialize execution pool
        self.semaphore = asyncio.Semaphore(max_workers)
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup handlers for system signals"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        self.logger.warning(f"Received signal {signum}")
        asyncio.create_task(self.shutdown())

    async def execute(self, 
                     task_id: str,
                     func: Callable,
                     args: tuple = (),
                     kwargs: dict = None,
                     timeout: Optional[int] = None) -> ExecutionResult:
        """Execute a task with resource monitoring"""
        start_time = datetime.now()
        kwargs = kwargs or {}
        
        try:
            async with self.semaphore:
                if not self._check_resources():
                    raise ResourceError("System resources exceeded limits")
                
                self.state = ExecutorState.RUNNING
                execution_task = asyncio.create_task(
                    self._monitored_execution(task_id, func, args, kwargs)
                )
                self.active_tasks[task_id] = execution_task
                
                try:
                    if timeout:
                        result = await asyncio.wait_for(execution_task, timeout)
                    else:
                        result = await execution_task
                        
                    runtime = (datetime.now() - start_time).total_seconds()
                    resources = self._get_resource_usage()
                    
                    execution_result = ExecutionResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        runtime=runtime,
                        resources=resources,
                        metadata={
                            'start_time': start_time,
                            'end_time': datetime.now(),
                            'executor_id': id(self)
                        }
                    )
                    
                except asyncio.TimeoutError:
                    raise ExecutionError(f"Task {task_id} timed out after {timeout} seconds")
                    
                finally:
                    self.active_tasks.pop(task_id, None)
                    
                self.results[task_id] = execution_result
                return execution_result
                
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            
            execution_result = ExecutionResult(
                task_id=task_id,
                success=False,
                result=None,
                error=error_msg,
                runtime=(datetime.now() - start_time).total_seconds(),
                resources=self._get_resource_usage()
            )
            
            self.results[task_id] = execution_result
            self.state = ExecutorState.ERROR
            return execution_result

    async def _monitored_execution(self, 
                                 task_id: str, 
                                 func: Callable, 
                                 args: tuple, 
                                 kwargs: dict) -> Any:
        """Execute task with resource monitoring"""
        monitor_task = asyncio.create_task(self._monitor_resources(task_id))
        
        try:
            result = await func(*args, **kwargs)
            return result
            
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_resources(self, task_id: str):
        """Monitor resource usage during task execution"""
        while True:
            usage = self._get_resource_usage()
            self.resource_usage.append({
                'timestamp': datetime.now(),
                'task_id': task_id,
                'usage': usage
            })
            
            if not self._check_resources():
                self.logger.warning(f"Resource limits exceeded for task {task_id}")
                
            await asyncio.sleep(1)  # Monitor every second

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        return {
            'cpu_percent': self.process.cpu_percent(),
            'memory_percent': self.process.memory_percent(),
            'num_threads': self.process.num_threads(),
            'num_fds': self.process.num_fds()
        }

    def _check_resources(self) -> bool:
        """Check if resource usage is within limits"""
        usage = self._get_resource_usage()
        
        return (
            usage['cpu_percent'] <= self.resource_limits['cpu_percent'] and
            usage['memory_percent'] <= self.resource_limits['memory_percent']
        )

    async def shutdown(self):
        """Gracefully shutdown the executor"""
        self.state = ExecutorState.STOPPED
        
        # Cancel all active tasks
        for task_id, task in self.active_tasks.items():
            self.logger.info(f"Cancelling task {task_id}")
            task.cancel()
            
        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        self.logger.info("Executor shutdown complete")

    def get_result(self, task_id: str) -> Optional[ExecutionResult]:
        """Get the result of a specific task"""
        return self.results.get(task_id)

    def get_resource_history(self, task_id: Optional[str] = None) -> List[Dict]:
        """Get resource usage history"""
        if task_id:
            return [entry for entry in self.resource_usage if entry['task_id'] == task_id]
        return self.resource_usage

class ExecutionError(Exception):
    pass

class ResourceError(Exception):
    pass

if __name__ == "__main__":
    # Example usage
    async def main():
        executor = TaskExecutor(max_workers=3)
        
        # Example async task
        async def example_task(name: str, delay: int = 1):
            await asyncio.sleep(delay)
            return f"Task {name} completed"
        
        # Execute tasks
        results = []
        for i in range(5):
            result = await executor.execute(
                f"task_{i}",
                example_task,
                args=(f"test_{i}",),
                kwargs={"delay": i}
            )
            results.append(result)
            
        # Print results
        for result in results:
            print(f"Task {result.task_id}: Success={result.success}, "
                  f"Runtime={result.runtime:.2f}s")
            
        # Get resource usage
        usage = executor.get_resource_history()
        print(f"Resource usage history: {len(usage)} entries")
        
        await executor.shutdown()

    asyncio.run(main())