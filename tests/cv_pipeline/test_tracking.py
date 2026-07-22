"""Production tracking websocket contract registration."""

import asyncio
import threading

import numpy as np

from backend.cv_pipeline import main

app = main.app


def test_tracking_websocket_route_is_registered():
    assert any(route.path == "/tracking/{session_id}" for route in app.routes)


def test_stale_stream_teardown_does_not_stop_replacement_worker():
    session_id = "generation-test"
    main.active_workers.add(session_id)
    main.stream_generations[session_id] = 2
    try:
        asyncio.run(main._teardown_stream(session_id, stream_generation=1))
        assert session_id in main.active_workers
        assert main.stream_generations[session_id] == 2
    finally:
        main.active_workers.discard(session_id)
        main.stream_generations.pop(session_id, None)


def test_worker_generation_rejects_old_worker():
    session_id = "worker-generation-test"
    main.active_workers.add(session_id)
    main.worker_generations[session_id] = 3
    try:
        assert main._worker_is_current(session_id, 3) is True
        assert main._worker_is_current(session_id, 2) is False
    finally:
        main.active_workers.discard(session_id)
        main.worker_generations.pop(session_id, None)


class _FakeExtractor:
    def __init__(self, session_id, *, fail=False):
        self.session_id = session_id
        self.fail = fail
        self.closed = False

    def extract(self, frame):
        if self.fail:
            raise RuntimeError("transient detector failure")
        return main.session_manager.get_or_create(self.session_id).last_raw

    def close(self):
        self.closed = True


def test_worker_generations_are_not_reused_after_replacement(monkeypatch):
    session_id = "worker-replacement-test"
    created = []

    def make_extractor(sid):
        extractor = _FakeExtractor(sid)
        created.append(extractor)
        return extractor

    monkeypatch.setattr(main, "SignalExtractor", make_extractor)

    async def scenario():
        await main._ensure_worker_running(session_id)
        first = main.worker_tasks[session_id]
        assert len(first.tasks) == 4
        await main._await_worker(main._retire_worker(session_id))

        await main._ensure_worker_running(session_id)
        second = main.worker_tasks[session_id]
        assert second.generation > first.generation
        assert second is not first
        await main._await_worker(main._retire_worker(session_id))

    asyncio.run(scenario())
    assert [extractor.closed for extractor in created] == [True, True]
    main.session_manager.remove(session_id)


def test_detector_failure_cleans_active_worker(monkeypatch):
    session_id = "worker-failure-test"
    extractor = _FakeExtractor(session_id, fail=True)
    monkeypatch.setattr(main, "SignalExtractor", lambda sid: extractor)

    async def scenario():
        await main._ensure_worker_running(session_id)
        group = main.worker_tasks[session_id]
        group.slot.put(np.zeros((2, 2, 3), dtype=np.uint8))
        await asyncio.wait_for(group.processing_task, timeout=1.0)
        assert session_id not in main.active_workers
        assert session_id not in main.worker_tasks
        await main._ensure_worker_running(session_id)
        replacement = main.worker_tasks[session_id]
        assert replacement.generation > group.generation
        await main._await_worker(main._retire_worker(session_id))

    asyncio.run(scenario())
    assert extractor.closed is True
    main.session_manager.remove(session_id)


def test_extractor_is_not_closed_during_inference(monkeypatch):
    session_id = "worker-close-ownership-test"
    inference_started = threading.Event()
    release_inference = threading.Event()

    class BlockingExtractor(_FakeExtractor):
        def extract(self, frame):
            inference_started.set()
            release_inference.wait(timeout=1.0)
            return super().extract(frame)

    extractor = BlockingExtractor(session_id)
    monkeypatch.setattr(main, "SignalExtractor", lambda sid: extractor)

    async def scenario():
        await main._ensure_worker_running(session_id)
        group = main.worker_tasks[session_id]
        group.slot.put(np.zeros((2, 2, 3), dtype=np.uint8))
        assert await asyncio.to_thread(inference_started.wait, 1.0)

        cleanup = asyncio.create_task(
            main._await_worker(main._retire_worker(session_id))
        )
        await asyncio.sleep(0)
        assert extractor.closed is False
        release_inference.set()
        await asyncio.wait_for(cleanup, timeout=1.0)

    asyncio.run(scenario())
    assert extractor.closed is True
    main.session_manager.remove(session_id)


def test_gc_cleanup_does_not_remove_reconnect_created_during_await():
    session_id = "gc-reconnect-test"
    old_extractor = _FakeExtractor(session_id)
    new_extractor = _FakeExtractor(session_id)
    new_slot = main.FrameSlot()
    replacement_socket = object()

    class ReconnectingSocket:
        async def close(self, **kwargs):
            main.session_manager.get_or_create(session_id)
            main.tracking_subscribers[session_id] = {replacement_socket}
            main.extractors[session_id] = new_extractor
            main.frame_slots[session_id] = new_slot
            await asyncio.sleep(0)

    main.extractors[session_id] = old_extractor
    main.frame_slots[session_id] = main.FrameSlot()
    main.tracking_subscribers[session_id] = {ReconnectingSocket()}
    main.session_manager.remove(session_id)

    asyncio.run(main._cleanup_removed_session(session_id))

    assert main.session_manager.get(session_id) is not None
    assert main.tracking_subscribers[session_id] == {replacement_socket}
    assert main.extractors[session_id] is new_extractor
    assert main.frame_slots[session_id] is new_slot
    assert old_extractor.closed is True

    main.tracking_subscribers.pop(session_id, None)
    main.extractors.pop(session_id, None)
    main.frame_slots.pop(session_id, None)
    main.session_manager.remove(session_id)
