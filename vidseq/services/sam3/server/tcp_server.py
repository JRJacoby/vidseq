"""
SAM3 TCP Server Worker.

Runs as standalone TCP server process. Accepts multiple connections,
queues all commands, processes them sequentially for GPU safety.
"""

import json
import os
import queue
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from vidseq.models.registry import Project
from vidseq.models.video import Video
from vidseq.services.database_manager import DatabaseManager
from vidseq.services.sam3.server.commands import (
    handle_add_prompt,
    handle_close_session,
    handle_generate_training_masks,
    handle_init_session,
    handle_load_model,
    handle_propagate,
    handle_refine_mask,
    handle_reset_frame,
    handle_reset_video,
    handle_segment_videos_batch,
    handle_shutdown,
)
from vidseq.services.sam3.config import (
    cleanup_port_files,
    find_free_port,
    write_pid_file,
    write_port_file,
)
from vidseq.services.model_manager import ensure_sam3_model


class SAM3TCPServer:
    """TCP server for SAM3 worker with sequential command processing."""

    def __init__(self, port: Optional[int] = None):
        """
        Initialize TCP server.

        Args:
            port: Port to bind to. If None, finds free port automatically.
        """
        self.port = port or find_free_port()
        self.server_socket: Optional[socket.socket] = None
        self.command_queue: queue.Queue = queue.Queue()
        self.active_connections: set[socket.socket] = set()
        self.connection_lock = threading.Lock()
        self.shutdown_timer: Optional[threading.Timer] = None
        self.shutdown_timeout = 10.0
        self.running = False
        self._is_processing = False

        # SAM3 state - StreamingSegmentor manages sessions internally
        self._segmentor = None

        # Config paths
        self.checkpoint_path: Optional[Path] = None  # Set in start() after ensuring model exists

    def _recover_stale_in_progress_states(self) -> None:
        """Reset any 'in_progress' videos from previous crashed runs."""
        try:
            db_manager = DatabaseManager.get_instance()

            # Get all project databases from registry
            registry_engine = db_manager.get_registry_engine()
            with Session(registry_engine) as session:
                projects = session.execute(select(Project)).scalars().all()

                recovered_count = 0
                for project in projects:
                    project_path = Path(project.path)
                    if not project_path.exists():
                        continue

                    try:
                        project_engine = db_manager.get_project_engine(project_path)
                        with Session(project_engine) as proj_session:
                            # Find all videos with segmentation_status='in_progress'
                            result = proj_session.execute(
                                select(Video).where(Video.segmentation_status == "in_progress")
                            )
                            in_progress_videos = result.scalars().all()

                            if in_progress_videos:
                                # Reset them to None
                                for video in in_progress_videos:
                                    proj_session.execute(
                                        update(Video)
                                        .where(Video.id == video.id)
                                        .values(segmentation_status=None)
                                    )
                                    recovered_count += 1

                                proj_session.commit()
                                print(
                                    f"[SAM3 Worker] Recovered {len(in_progress_videos)} stale "
                                    f"'in_progress' videos in project {project.name}"
                                )
                    except Exception as e:
                        print(f"[SAM3 Worker] Warning: Failed to recover project {project.name}: {e}")
                        continue

                if recovered_count > 0:
                    print(f"[SAM3 Worker] Total recovered videos: {recovered_count}")
        except Exception as e:
            print(f"[SAM3 Worker] Warning: Failed to recover stale states: {e}")

    def start(self) -> None:
        """Start the TCP server."""
        # Ensure model checkpoint is downloaded before starting
        print("[SAM3 Worker] Ensuring model checkpoint is available...")
        self.checkpoint_path = ensure_sam3_model()
        print(f"[SAM3 Worker] Model checkpoint: {self.checkpoint_path}")

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("localhost", self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        self.running = True

        print(f"[SAM3 Worker] TCP server started on localhost:{self.port}")

        # Write port and PID files
        write_port_file(self.port)
        write_pid_file(os.getpid())

        # Recover stale 'in_progress' states from previous crashes
        self._recover_stale_in_progress_states()

        # Start command processing thread
        processing_thread = threading.Thread(target=self._process_commands, daemon=True)
        processing_thread.start()

        # Accept connections
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                # Only log on first connection or when connection count changes significantly
                with self.connection_lock:
                    conn_count = len(self.active_connections)
                if conn_count == 0:
                    print(f"[SAM3 Worker] First client connected from {addr}")
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True,
                )
                client_thread.start()
            except socket.timeout:
                # Timeout - check if we should still be running
                continue
            except OSError:
                # Server socket closed
                break

    def _handle_client(self, conn: socket.socket, addr: tuple) -> None:
        """Handle a client connection."""
        with self.connection_lock:
            self.active_connections.add(conn)
            conn_count = len(self.active_connections)
            # Cancel any pending shutdown
            if self.shutdown_timer:
                self.shutdown_timer.cancel()
                self.shutdown_timer = None

        # Only log if this is the first connection
        if conn_count == 1:
            print(f"[SAM3 Worker] Client connected from {addr}")

        try:
            # Set a short timeout for initial read to detect quick disconnects
            conn.settimeout(1.0)

            while self.running:
                # Receive command length
                length_bytes = self._recv_exact(conn, 4)
                if len(length_bytes) != 4:
                    break

                # Once we receive a command, remove timeout (command processing may take time)
                conn.settimeout(None)

                cmd_length = struct.unpack(">I", length_bytes)[0]

                # Receive command data
                cmd_bytes = self._recv_exact(conn, cmd_length)
                cmd_json = cmd_bytes.decode("utf-8")
                cmd = json.loads(cmd_json)

                # Create response callback
                response_callback = lambda result: self._send_response(conn, result)

                # Add to command queue
                self.command_queue.put((cmd, response_callback))
        except socket.timeout:
            # Connection opened but closed quickly (likely health check)
            pass
        except Exception as e:
            # Only log actual errors, not quick disconnects
            if "timed out" not in str(e).lower():
                print(f"[SAM3 Worker] Error handling client {addr}: {e}")
        finally:
            # Connection closed
            with self.connection_lock:
                was_last = len(self.active_connections) == 1
                self.active_connections.discard(conn)
                # If no connections left, schedule shutdown
                if len(self.active_connections) == 0:
                    if was_last:
                        print(f"[SAM3 Worker] Last client disconnected from {addr}")
                    self._schedule_shutdown()
            try:
                conn.close()
            except Exception:
                pass

    def _recv_exact(self, conn: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b""
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                return data
            data += chunk
        return data

    def _send_response(self, conn: socket.socket, result: dict) -> None:
        """Send response to client."""
        # Check if connection is still active
        with self.connection_lock:
            if conn not in self.active_connections:
                return  # Connection closed, skip sending

        try:
            result_json = json.dumps(result)
            result_bytes = result_json.encode("utf-8")

            # Send length prefix
            length_prefix = struct.pack(">I", len(result_bytes))
            conn.sendall(length_prefix)

            # Send response data
            conn.sendall(result_bytes)
        except Exception as e:
            # Connection may have closed, remove it
            with self.connection_lock:
                self.active_connections.discard(conn)
            print(f"[SAM3 Worker] Error sending response: {e}")

    def _process_commands(self) -> None:
        """Process commands sequentially from queue."""
        while self.running:
            try:
                cmd, response_callback = self.command_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._is_processing = True
                self._handle_command(cmd, response_callback)
            except Exception as e:
                print(f"[SAM3 Worker] Error processing command: {e}")
                import traceback

                traceback.print_exc()
                error_result = {
                    "type": "error",
                    "error": str(e),
                    "request_id": cmd.get("request_id"),
                }
                response_callback(error_result)
            finally:
                self._is_processing = False
                # If no connections, schedule shutdown now that we're done processing
                with self.connection_lock:
                    if len(self.active_connections) == 0:
                        self._schedule_shutdown()

    def _handle_command(self, cmd: dict, response_callback) -> None:
        """Route command to appropriate handler."""
        cmd_type = cmd.get("type")
        request_id = cmd.get("request_id")

        try:
            if cmd_type == "load_model":
                if self.checkpoint_path is None:
                    raise RuntimeError("Checkpoint path not set")
                result, self._segmentor = handle_load_model(self.checkpoint_path)

            elif cmd_type == "init_session":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_init_session(cmd, self._segmentor)

            elif cmd_type == "add_prompt":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_add_prompt(cmd, self._segmentor)

            elif cmd_type == "refine_mask":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_refine_mask(cmd, self._segmentor)

            elif cmd_type == "propagate":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_propagate(cmd, self._segmentor)

            elif cmd_type == "generate_training_masks":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_generate_training_masks(cmd, self._segmentor)

            elif cmd_type == "reset_frame":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_reset_frame(cmd, self._segmentor)

            elif cmd_type == "reset_video":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_reset_video(cmd, self._segmentor)

            elif cmd_type == "segment_videos_batch":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_segment_videos_batch(
                    cmd,
                    self._segmentor,
                    response_callback,
                )

            elif cmd_type == "close_session":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_close_session(cmd, self._segmentor)

            elif cmd_type == "shutdown":
                if self._segmentor is None:
                    raise RuntimeError("Model not loaded")
                result = handle_shutdown(self._segmentor)
                self.running = False

            else:
                print(f"[SAM3 Worker] Unknown command type: {cmd_type}")
                result = {
                    "type": "error",
                    "error": f"Unknown command type: {cmd_type}",
                }

            # Add request_id to result if present
            if request_id is not None:
                result["request_id"] = request_id

            response_callback(result)

        except Exception as e:
            print(f"[SAM3 Worker] Error in command {cmd_type}: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "type": f"{cmd_type}_result" if cmd_type else "error",
                "status": "error",
                "error": str(e),
            }
            if request_id is not None:
                result["request_id"] = request_id
            response_callback(result)

    def _schedule_shutdown(self) -> None:
        """Schedule shutdown if no connections arrive within timeout."""

        def shutdown_if_still_empty():
            time.sleep(self.shutdown_timeout)
            with self.connection_lock:
                if len(self.active_connections) == 0 and not self._is_processing:
                    print("[SAM3 Worker] No connections and not processing, shutting down...")
                    self.stop()
                    print("[SAM3 Worker] Exiting process...")
                    os._exit(0)

        if self.shutdown_timer:
            self.shutdown_timer.cancel()
        self.shutdown_timer = threading.Timer(self.shutdown_timeout, shutdown_if_still_empty)
        self.shutdown_timer.start()

    def stop(self) -> None:
        """Stop the server."""
        import torch

        self.running = False

        # Close all active connections
        with self.connection_lock:
            for conn in list(self.active_connections):
                try:
                    conn.close()
                except Exception:
                    pass
            self.active_connections.clear()

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass

        # Free GPU memory by deleting segmentor and clearing CUDA cache
        if self._segmentor is not None:
            del self._segmentor
            self._segmentor = None
            torch.cuda.empty_cache()
            print("[SAM3 Worker] GPU memory cleared")

        cleanup_port_files()


def main():
    """Main entry point for standalone server."""
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 TCP Worker Server")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: auto-assign)",
    )
    args = parser.parse_args()

    server = SAM3TCPServer(port=args.port)

    def signal_handler(signum, frame):
        print("\n[SAM3 Worker] Received shutdown signal, cleaning up...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[SAM3 Worker] Interrupted, shutting down...")
        server.stop()
    finally:
        cleanup_port_files()


if __name__ == "__main__":
    main()
