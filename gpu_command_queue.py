"""
A server-client interface for managing large batch JAX jobs on a single machine with multiple GPUs.

First, start the server:

```
python gpu_command_queue.py --gpus 7 --memfrac-per-proc 0.95
```

Then, inside another script, you can use the `GPUQueueClient` to send commands to the server (which by default ALWAYS runs on port 5034):

```
from gpu_command_queue import GPUQueueClient
queue = GPUQueueClient(git_commit_id="most_recent")

# Send commands as a list of (command, stdout_file) tuples. Stdoutfile can be "" for no saving of stdout logging.
commands = [
    ("python -c 'import time; import jax.numpy as jnp; a=jnp.ones(5); time.sleep(1); print(\"I SLEPT HAPPY 1\")'", "testlogs/v1.txt"),
    ("python -c 'import time; import jax.numpy as jnp; a=jnp.ones(5); time.sleep(2); print(\"I SLEPT HAPPY 2\")'", ""),
    ("python -c 'import time; import jax.numpy as jnp; a=jnp.ones(5); time.sleep(3); print(\"I SLEPT HAPPY 3\")'", "testlogs/v3.txt"),
]

if queue.git_repo.is_dirty():
    user_input = input("Repo has uncommitted changes: continue? [Y/n] ").strip().lower()
    if user_input == 'n':
        raise ValueError("Aborting due to uncommitted changes in the repository.") 

queue.send_command_list(cmd_tuples)
```

Kill the server with

```
fuser -k 5034/tcp
```
"""
#%%
from pydantic import BaseModel
from pathlib import Path
from typing import Union, Dict, List
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import queue
import threading
import subprocess
import os
from typing import List, Union, Tuple
import psutil
import tyro
from tyro.conf import Positional
from datetime import datetime
import math
import requests
import tempfile
import git
from pathlib import Path
from typing import List, Optional
from urllib3.exceptions import NewConnectionError
from requests.exceptions import ConnectionError
import os

GPU_SERVER_HOST = "0.0.0.0"
GPU_SERVER_PORT = 5034

class CommandMessage(BaseModel):
    command: str = f""" python -c "import time; time.sleep(1); print('I SLEPT HAPPY')" """
    stdout_file: str = ""
    EXPLOGGER_ROOT: str = str(Path.cwd())
    EXPLOGGER_GITDIR: str = str(Path.cwd())
    working_dir: str = str(Path.cwd())

class GPUQueueServer:
    """
    Usage:
        server = GPUQueueServer(gpus=4, memfrac_per_proc=0.45)
        server.run()

    kill server from CLI with 
        fuser -k 5034/tcp
    """
    def __init__(self, 
                 gpus: Positional[Union[int, List[int]]], # List of GPU IDs to use. If an integer, just takes the first N gpus
                 memfrac_per_proc: float = 0.45 # How much memory per GPU to use
                 ):
        self.gpu_threads = []
        if isinstance(gpus, int):
            self.num_gpus = gpus
            self.gpus = list(range(self.num_gpus))
        elif isinstance(gpus, list):
            self.num_gpus = len(gpus)
            self.gpus = gpus
        else:
            raise ValueError("gpus must be an integer or a list of integers.")
            
        self.memfrac_per_proc = memfrac_per_proc
        self.procs_per_gpu = math.floor(1 / memfrac_per_proc)
        self.total_procs = self.num_gpus * self.procs_per_gpu
        self.gpus = gpus if isinstance(gpus, list) else list(range(gpus))
        self.stop_event = threading.Event()
        self.processes: Dict[int, subprocess.Popen] = {}
        self.process_lock = threading.Lock()
        self.memfrac_per_proc = memfrac_per_proc
        self.command_queue = queue.Queue()
        self.gpu_threads = self.start_gpu_threads()
        
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/enqueue")
        async def enqueue_message(msg: Union[CommandMessage, List[CommandMessage]], background_tasks: BackgroundTasks):
            background_tasks.add_task(self.enqueue_message, msg)
            nmsgs = len(msg) if isinstance(msg, list) else 1
            return {"status": f"{nmsgs} message(s) enqueued"}

        @self.app.get("/stop-and-restart-threads")
        async def stop_and_restart_threads():
            self.stop_threads()
            self.gpu_threads = self.start_gpu_threads()
            return {"status": "Threads stopped and restarted"}

        @self.app.get("/healthy-gpu-queue")
        async def stop_and_restart_threads():
            return {"status": "Yep, I'm operating"}

        # test this
        @self.app.get("/queue-length")
        async def queue_length():
            return self.command_queue.qsize()

        # Delete item from queue

    def start_gpu_threads(self):
        if hasattr(self, "gpu_threads"):
            self.stop_threads()
        self.stop_event.clear()
        gpu_threads = []
        for i in range(self.total_procs):
            thread = threading.Thread(target=self._run_commands, args=(i,))
            thread.start()
            gpu_threads.append(thread)
        return gpu_threads


    def enqueue_message(self, msg: Union[CommandMessage, List[CommandMessage]]):
        if isinstance(msg, list):
            for m in msg:
                print(f"Enqueuing message: {m}")
                self.command_queue.put(m)
        else:
            self.command_queue.put(msg)

    def _run_commands(self, proc_id: int):
        gpu_idx = proc_id % self.num_gpus
        gpu_id = self.gpus[gpu_idx]

        while not self.stop_event.is_set():
            try:
                msg = self.command_queue.get(timeout=1)
                if msg is None:
                    break

                print(f"Running on GPU {gpu_id} with procid=`{proc_id}` in `{msg.working_dir}`: {msg.command}")
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(self.memfrac_per_proc)
                env['EXPLOGGER_ROOT'] = msg.EXPLOGGER_ROOT
                env['EXPLOGGER_GITDIR'] = msg.EXPLOGGER_GITDIR

                print(f"Running on GPU {gpu_id} in `{msg.working_dir}`: {msg.command}")
                delim = "\n" + "=."*40 + "\n"

                if msg.stdout_file:
                    Path(msg.stdout_file).parent.mkdir(parents=True, exist_ok=True)
                    print(f"Output will be redirected to: {msg.stdout_file}")
                    delim = "\n" + "=."*40 + "\n"
                    stdout = open(msg.stdout_file, 'a+')
                    stdout.write(delim + str(datetime.now()) + delim)
                    stderr = subprocess.STDOUT
                else:
                    stdout = None
                    stderr = None

                subprocess.run(
                    msg.command,
                    shell=True,
                    env=env,
                    cwd=msg.working_dir,
                    stdout=stdout,
                    stderr=stderr,
                    text=True
                )

                if msg.stdout_file:
                    stdout.close()

                self.command_queue.task_done()
            except queue.Empty:
                continue

    def stop_threads(self):
        self.stop_event.set()

        # Terminate all running processes
        with self.process_lock:
            for pid, process in self.processes.items():
                try:
                    parent = psutil.Process(pid)
                    for child in parent.children(recursive=True):
                        child.terminate()
                    parent.terminate()
                except psutil.NoSuchProcess:
                    pass

        # Wait for processes to terminate
        gone, alive = psutil.wait_procs(list(self.processes.values()), timeout=3)

        # If any processes are still alive, kill them
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass

        # Clear the processes dictionary
        with self.process_lock:
            self.processes.clear()

        # Clear the queue
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Queue.Empty:
                break

        # Join threads with timeout
        for thread in self.gpu_threads:
            thread.join(timeout=1)  # 5 seconds timeout

        # Check if any threads are still alive
        alive_threads = [t for t in self.gpu_threads if t.is_alive()]
        if alive_threads:
            print(f"Warning: {len(alive_threads)} threads did not terminate properly.")

        # Reset gpu_threads
        self.gpu_threads = []

        return True

    def run(self):
        uvicorn.run(self.app, host=GPU_SERVER_HOST, port=GPU_SERVER_PORT)

    def __del__(self):
        print("Cleaning up GPUQueueServer...")
        self.stop_threads()
        print("GPUQueueServer cleanup complete.")


class GPUQueueClient:
    def __init__(self, git_repo_path: str = ".", git_commit_id: Optional[str] = None):
        self.git_repo = git.Repo(git_repo_path, search_parent_directories=True)
        self.git_repo_dir = Path(self.git_repo.git_dir).parent
        self.git_commit_id = self.get_commit_id(git_commit_id)
        self.server_url = f"http://{GPU_SERVER_HOST}:{GPU_SERVER_PORT}"
        self.temp_dir = None
        self.setup_environment()
        assert self.is_server_running(), f"""Server is not running. You can start the server by running `python gpu_command_queue.py NGPUS`"""


    def setup_environment(self):
        if self.git_commit_id is not None:
            self.temp_dir = tempfile.mkdtemp()
            print(f"Creating temporary environment at: {self.temp_dir}")
            git.Repo.clone_from(self.git_repo_dir, self.temp_dir)
            repo = git.Repo(self.temp_dir)
            print("CID: ", self.git_commit_id)
            repo.git.checkout(self.git_commit_id)

            print(f"Checked out commit {self.git_commit_id} in temporary environment")
        else:
            print("No Git commit specified. Using current working directory.")

    def get_commit_id(self, commit_id: Optional[str]) -> Optional[str]:
        if commit_id == "most_recent":
            return self.git_repo.head.commit.hexsha
        return commit_id

    def command_to_message(self, cmd_tuple: Tuple[str, str]) -> CommandMessage:
        cmd, stdout_file = cmd_tuple
        return CommandMessage(
            command=cmd,
            stdout_file=stdout_file,
            EXPLOGGER_ROOT=str(os.getcwd()),
            EXPLOGGER_GITDIR=str(os.getcwd()),
            working_dir=self.temp_dir if self.temp_dir else str(os.getcwd())
        )

    def send_command(self, cmd_tuple: Tuple[str, str]):
        message = self.command_to_message(cmd_tuple)
        response = requests.post(f"{self.server_url}/enqueue", json=message.model_dump())
        if response.status_code != 200:
            print(f"Error sending command: {response.text}")

    def send_command_list(self, cmds: List[Tuple[str, str]]):
        messages = [self.command_to_message(cmd).model_dump() for cmd in cmds]
        response = requests.post(f"{self.server_url}/enqueue", json=messages)
        if response.status_code != 200:
            print(f"Error sending command: {response.text}")

        print(response.json()["status"])

    def is_server_running(self) -> bool:
        try:
            response = requests.get(f"{self.server_url}/healthy-gpu-queue")
            return response.status_code == 200
        except (ConnectionError, NewConnectionError):
            return False

# Usage:
if __name__ == "__main__":
    server = tyro.cli(GPUQueueServer)
    server.run()