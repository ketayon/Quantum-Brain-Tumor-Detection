import logging
import concurrent.futures

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class JobScheduler:
    """Manages Scheduling for Quantum & Classical Tasks"""
    
    def __init__(self, max_workers=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def schedule_task(self, task, *args, **kwargs):
        log.info("Scheduling Task: %s", task.__name__)
        future = self.executor.submit(task, *args, **kwargs)
        return future.result()