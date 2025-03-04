from dataclasses import dataclass
from datetime import datetime
import asyncio
import httpx
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    payload: Dict[str, Any]
    created_at: datetime
    future: asyncio.Future


class RequestBatcher:
    def __init__(
        self, batch_size: int = 25, max_wait_time: float = 0.05, max_retries: int = 2
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_retries = max_retries
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._http_client: Optional[httpx.AsyncClient] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize HTTP client and start worker"""
        self._http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=100, max_connections=200, keepalive_expiry=30
            ),
        )
        self._worker_task = asyncio.create_task(self._batch_worker())
        logger.info("RequestBatcher initialized with batch_size=%d", self.batch_size)

    async def _batch_worker(self) -> None:
        """Background worker for processing batches"""
        while not self._shutdown_event.is_set():
            try:
                batch = []
                # Get first request with timeout
                try:
                    request = await asyncio.wait_for(
                        self._request_queue.get(), timeout=self.max_wait_time
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    continue

                # Collect additional requests without blocking
                while len(batch) < self.batch_size:
                    try:
                        request = self._request_queue.get_nowait()
                        batch.append(request)
                    except asyncio.QueueEmpty:
                        break

                if batch:
                    await self._process_batch_requests(batch)

            except Exception as e:
                logger.error("Batch worker error: %s", str(e), exc_info=True)
                await asyncio.sleep(0.1)

    async def _process_batch_requests(self, batch: List[BatchRequest]) -> None:
        """Process a batch of requests"""
        try:
            # Prepare batch request
            first_request = batch[0].payload
            batch_payload = {
                "requests": [req.payload["request"] for req in batch],
                "url": first_request["url"],
                "headers": first_request.get("headers", {}),
                "timeout": first_request.get("timeout", 30.0),
            }

            # Make batch request with retries
            for attempt in range(self.max_retries):
                try:
                    response = await self._make_batch_request(batch_payload)

                    # Distribute results
                    for req, result in zip(batch, response["results"]):
                        if not req.future.done():
                            req.future.set_result(result)
                    return

                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(0.1 * (attempt + 1))

        except Exception as e:
            logger.error("Batch request failed: %s", str(e), exc_info=True)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    async def add_request(self, payload: Dict[str, Any]) -> Any:
        """Add request to queue and return future"""
        if self._shutdown_event.is_set():
            raise RuntimeError("RequestBatcher is shutting down")

        future = asyncio.Future()
        request = BatchRequest(
            payload=payload, created_at=datetime.now(), future=future
        )

        await self._request_queue.put(request)
        return await future

    async def shutdown(self) -> None:
        """Cleanup resources"""
        self._shutdown_event.set()

        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._worker_task.cancel()

        if self._http_client:
            await self._http_client.aclose()

        logger.info("RequestBatcher shutdown complete")
