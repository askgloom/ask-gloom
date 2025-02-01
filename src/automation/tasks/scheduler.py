import asyncio
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import uuid
from enum import Enum
import heapq
from contextlib import contextmanager

class TaskPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    id: str
    name: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority
    schedule_time: datetime
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    metadata: Dict = None

class TaskScheduler:
    def __init__(self, max_concurrent: int = 10):
        self.logger = logging.getLogger(__name__)
        self.max_concurrent = max_concurrent
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.priority_queue: List[tuple] = []  # (schedule_time, priority, task_id)
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # State management
        self.is_running = False
        self.task_history: List[Dict] = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Statistics
        self.stats = {
            'total_scheduled': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_cancelled': 0
        }

    async def schedule_task(self,
                          name: str,
                          func: Callable,
                          args: tuple = (),
                          kwargs: dict = None,
                          priority: TaskPriority = TaskPriority.MEDIUM,
                          delay: Optional[Union[int, float]] = None,
                          timeout: Optional[int] = None) -> str:
        """Schedule a new task"""
        task_id = str(uuid.uuid4())
        schedule_time = datetime.now() + timedelta(seconds=delay or 0)
        
        task = Task(
            id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            schedule_time=schedule_time,
            timeout=timeout,
            metadata={
                'created_at': datetime.now(),
                'scheduler_id': id(self)
            }
        )
        
        self.tasks[task_id] = task
        heapq.heappush(
            self.priority_queue,
            (schedule_time.timestamp(), priority.value, task_id)
        )
        
        self.stats['total_scheduled'] += 1
        self.logger.info(f"Scheduled task {name} with ID {task_id}")
        
        return task_id

    async def start(self):
        """Start the task scheduler"""
        self.is_running = True
        self.logger.info("Task scheduler started")
        
        try:
            while self.is_running:
                await self._process_next_task()
                await asyncio.sleep(0.1)  # Prevent CPU overload
                
        except Exception as e:
            self.logger.error(f"Scheduler error: {str(e)}")
            raise SchedulerError(f"Scheduler failed: {str(e)}")

    async def stop(self):
        """Stop the task scheduler"""
        self.is_running = False
        
        # Cancel all running tasks
        for task_id, task in self.running_tasks.items():
            task.cancel()
            self.tasks[task_id].status = TaskStatus.CANCELLED
            
        self.stats['total_cancelled'] += len(self.running_tasks)
        self.running_tasks.clear()
        
        self.logger.info("Task scheduler stopped")

    async def _process_next_task(self):
        """Process the next task in the queue"""
        if not self.priority_queue:
            return
            
        now = datetime.now().timestamp()
        next_time, _, task_id = self.priority_queue[0]
        
        if next_time <= now:
            heapq.heappop(self.priority_queue)
            task = self.tasks[task_id]
            
            if task.status == TaskStatus.PENDING:
                await self._execute_task(task)

    async def _execute_task(self, task: Task):
        """Execute a task with error handling and retries"""
        async with self.semaphore:
            task.status = TaskStatus.RUNNING
            
            try:
                if task.timeout:
                    async with asyncio.timeout(task.timeout):
                        task.result = await task.func(*task.args, **task.kwargs)
                else:
                    task.result = await task.func(*task.args, **task.kwargs)
                    
                task.status = TaskStatus.COMPLETED
                self.stats['total_completed'] += 1
                
            except Exception as e:
                task.error = str(e)
                
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    # Reschedule with exponential backoff
                    delay = 2 ** task.retry_count
                    new_schedule_time = datetime.now() + timedelta(seconds=delay)
                    heapq.heappush(
                        self.priority_queue,
                        (new_schedule_time.timestamp(), task.priority.value, task.id)
                    )
                else:
                    task.status = TaskStatus.FAILED
                    self.stats['total_failed'] += 1
                    self.logger.error(f"Task {task.name} failed: {str(e)}")

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a specific task"""
        task = self.tasks.get(task_id)
        return task.status if task else None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled or running task"""
        task = self.tasks.get(task_id)
        if not task:
            return False
            
        if task.status == TaskStatus.RUNNING:
            running_task = self.running_tasks.get(task_id)
            if running_task:
                running_task.cancel()
                self.stats['total_cancelled'] += 1
                
        task.status = TaskStatus.CANCELLED
        return True

    async def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        return {
            **self.stats,
            'current_queue_size': len(self.priority_queue),
            'running_tasks': len(self.running_tasks)
        }

    @contextmanager
    def task_context(self, task_id: str):
        """Context manager for task execution"""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        try:
            yield task
        finally:
            self.task_history.append({
                'task_id': task_id,
                'name': task.name,
                'status': task.status,
                'completed_at': datetime.now()
            })

class SchedulerError(Exception):
    pass

if __name__ == "__main__":
    # Example usage
    async def main():
        scheduler = TaskScheduler(max_concurrent=5)
        
        # Example async task
        async def example_task(name: str, delay: int = 1):
            await asyncio.sleep(delay)
            return f"Task {name} completed"
        
        # Schedule some tasks
        task_id = await scheduler.schedule_task(
            "example",
            example_task,
            args=("test",),
            kwargs={"delay": 2},
            priority=TaskPriority.HIGH
        )
        
        # Start scheduler
        scheduler_task = asyncio.create_task(scheduler.start())
        
        # Wait for a bit
        await asyncio.sleep(5)
        
        # Check task status
        status = await scheduler.get_task_status(task_id)
        print(f"Task status: {status}")
        
        # Get stats
        stats = await scheduler.get_stats()
        print(f"Scheduler stats: {stats}")
        
        # Stop scheduler
        await scheduler.stop()

    asyncio.run(main())