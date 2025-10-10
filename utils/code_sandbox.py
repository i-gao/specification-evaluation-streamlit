from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SandboxError
import re
from typing import List, Tuple
import uuid
import json
import os
import subprocess
import warnings

PROJ_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # utils/ -> specification-benchmark/


def _warn_unsandboxed():
    """Helper function to warn about running code without sandboxing."""
    warnings.warn(
        "⚠️  WARNING: Running code without Docker sandboxing! "
        "This means the code will execute in your current environment and could potentially "
        "access your files, network, or cause other security issues. Only use this option "
        "with trusted code or in controlled environments.",
        UserWarning,
        stacklevel=3,
    )


def execute_python_cmd_docker(
    command: str,
    docker_container_id: str = None,
    docker_image: str = None,
    session_timeout: int = 300,
    root_dir: str = "/sandbox",
    output_filenames: List[str] = [],
    input_filenames: List[str] = [],
) -> Tuple[str, List[str]]:
    """
    Executes a command in the sandbox environment.

    Args:
        command: The command to run.
        docker_container_id: The docker container id to use.
        output_filenames: The filenames to copy from the sandbox.
        input_filenames: The filenames to copy to the sandbox.

    ------- configs for if docker_container_id is not provided -------
        docker_image: The docker image to use if docker_container_id is not provided.
        session_timeout: The timeout for the session.
        root_dir: The root directory in the sandbox.

    Returns:
        The stdout of the command, and the output filenames.
    """
    assert (docker_image is not None) or (docker_container_id is not None), (
        "Either docker_image or docker_container_id must be provided"
    )
    with SandboxSession(
        container_id=docker_container_id,
        image=docker_image,
        runtime_configs={"auto_remove": True},
        session_timeout=session_timeout,
        commit_container=False,  # Prevent layer accumulation
    ) as sandbox_session:
        try:
            # Copy input files to sandbox
            for input_filename in input_filenames:
                sandbox_session.copy_to_runtime(
                    input_filename,
                    f"{root_dir}/{os.path.basename(input_filename)}",
                )

            result = sandbox_session.execute_command(command)
            exit_code = result.exit_code or 0
            assert exit_code == 0, (
                f"Exit code is nonzero: {result.text()} {getattr(result, 'stderr', '')}"
            )

            # Copy output files if specified
            copied_files = []
            for output_filename in output_filenames:
                sandbox_session.copy_from_runtime(
                    f"{root_dir}/{output_filename}",
                    output_filename,
                )
                copied_files.append(output_filename)
        except SandboxError:
            raise SandboxError("Something went wrong with the sandbox environment.")
        except Exception as exc:
            raise exc
    return result.text(), copied_files


def execute_python_cmd_local(
    command: str,
    session_timeout: int = 300,
    root_dir: str = ".",
    output_filenames: List[str] = [],
    input_filenames: List[str] = [],
    use_reliability_guard: bool = True,
    **kwargs,
) -> Tuple[str, List[str]]:
    """
    Executes a command in the local environment without Docker sandboxing.
    This is less secure and should be used with caution.

    Args:
        command: The command to run.
        session_timeout: The timeout for the session (in seconds).
        output_filenames: The filenames to copy from the execution.
        input_filenames: The filenames to copy to the execution (not used in local mode).
        root_dir: The root directory (not used in local mode).

    Returns:
        The stdout of the command, and the output filenames.
    """
    _warn_unsandboxed()

    # Use reliability guard as context manager if enabled
    try:
        # Helper to resolve file paths relative to the intended working directory
        def _resolve_local_path(path: str) -> str:
            if os.path.isabs(path):
                return path
            # If a root_dir is provided, resolve relative to it; otherwise use as-is
            return os.path.join(root_dir, path) if (root_dir is not None and root_dir != "") else path

        # Check files exist (resolved relative to the working directory that will be used)
        for input_filename in input_filenames:
            resolved_input = _resolve_local_path(input_filename)
            if not os.path.exists(resolved_input):
                raise Exception(
                    f"Input file not found: {resolved_input}"
                )

        # Prepare environment: protect project root from writes inside the cell runner
        env = os.environ.copy()
        env["READONLY_DIRS"] = PROJ_ROOT

        # Run the command with timeout, setting cwd to root_dir if provided
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=session_timeout,
            shell=True,
            env=env,
            cwd=(root_dir if (root_dir is not None and root_dir != "") else None),
        )

        exit_code = result.returncode or 0
        assert exit_code == 0, f"Exit code is nonzero: {result.stderr}"

        if len(result.stderr) > 0:
            raise Exception(result.stderr)

        # Check output files exist
        for output_filename in output_filenames:
            resolved_output = _resolve_local_path(output_filename)
            if not os.path.exists(resolved_output):
                raise Exception(
                    f"Output file not found: {resolved_output}"
                )

        # For local execution, we don't actually copy files, just return the names
        # This maintains compatibility with the Docker interface
        copied_files = output_filenames

        return result.stdout, copied_files

    except subprocess.TimeoutExpired:
        raise Exception(f"Command timed out after {session_timeout} seconds")
    except Exception as e:
        raise Exception(f"Command execution failed: {str(e)}")


def run_python_script_with_json_input(
    input_dict: dict,
    command: str,
    docker_container_id: str = None,
    docker_image: str = None,
    session_timeout: int = 300,
    input_filename: str = "_input_{uuid}.json",
    output_filenames: List[str] = [],
    root_dir: str = "/sandbox",
    **kwargs,
) -> Tuple[str, str, List[str]]:
    """
    Saves the input_dict to a json file, uploads it to the sandbox, and runs the command with the copied input file as the input.

    command, input_filename and output_filenames should be f-strings. Available placeholders are {uuid}, {input_filename} and the **kwargs.

    Args:
        input_dict: The input to the command.
        command: The command to run.
        docker_container_id: The docker container id to use.
        docker_image: The docker image to use if docker_container_id is not provided.
        session_timeout: The timeout for the session.
        input_filename: The filename to save the input to.
        output_filenames: The filenames to copy from the sandbox.
        kwargs: Additional arguments to pass to the command.

    Returns:
        The uuid of the run, the stdout of the command, and the output filenames
    """
    session_id = str(uuid.uuid4())
    formatted_input_filename = input_filename.format(uuid=session_id, **kwargs)

    # Determine where to place the input file
    if (docker_image is None) and (docker_container_id is None):
        # Local mode: write inside root_dir if provided, else current directory
        input_path = (
            os.path.join(root_dir, formatted_input_filename)
            if (root_dir is not None and root_dir != "")
            else formatted_input_filename
        )
    else:
        # Docker mode: write locally; it will be copied into the container
        input_path = formatted_input_filename

    json.dump(input_dict, open(input_path, "w"))

    try:
        # Format the command with placeholders
        formatted_command = command.format(
            input_filename=formatted_input_filename, uuid=session_id, **kwargs
        )

        # Format output filenames
        formatted_output_filenames = [
            output_filename.format(
                uuid=session_id, input_filename=formatted_input_filename, **kwargs
            )
            for output_filename in output_filenames
        ]

        # Call execute_python_cmd with the formatted command
        execute_fn = (
            execute_python_cmd_docker
            if (docker_image is not None) or (docker_container_id is not None)
            else execute_python_cmd_local
        )
        result, copied_files = execute_fn(
            command=formatted_command,
            docker_container_id=docker_container_id,
            docker_image=docker_image,
            session_timeout=session_timeout,
            root_dir=root_dir,
            output_filenames=formatted_output_filenames,
            input_filenames=[formatted_input_filename if (docker_image is not None or docker_container_id is not None) else input_path],
            **kwargs,
        )
    finally:
        try:
            os.remove(input_path)
        except Exception:
            pass

    return session_id, result, copied_files


CELL_SEPARATOR = "-" * 29


def reset_jupyter_session(
    filename: str,
) -> Tuple[str, str]:
    """
    Deletes the session file.
    Note that this does not restore files in the sandbox
    to their original state.
    """
    if os.path.exists(filename):
        os.remove(filename)


LOCAL_JUPYTER_SCRIPT = os.path.join(
    PROJ_ROOT, "utils/jupyter_docker_image/run_jupyter_cell.py"
)


def run_jupyter_script(
    filename: str,
    cell_code: str,
    docker_container_id: str = None,
    docker_image: str = None,
    session_timeout: int = 300,
    root_dir: str = "/sandbox",
) -> Tuple[str, str]:
    """
    Runs a jupyter cell in the sandbox environment.
    The docker image needs to include the run_jupyter_cell.py script. See utils/jupyter_docker_image as an example.

    BUGS: String literals with newlines will cause the cell to fail.

    Args:
        filename: The filename to write the cell code to.
        cell_code: The jupyter cell code to execute.
        docker_container_id: The docker container id to use.
        docker_image: The docker image to use if docker_container_id is not provided.
        session_timeout: The timeout for the session.
        root_dir: The root directory in the sandbox.
        kwargs: Additional arguments to pass to the command.

    Returns:
        The stdout of the command.
    """

    def _run_cell_code(cell_code):
        # Write the cell code to a file (do not inject cwd hacks)
        with open(filename, "a") as f:
            f.write("\n" + CELL_SEPARATOR + "\n" + cell_code)

        base_filename = os.path.basename(filename)
        script_path = (
            LOCAL_JUPYTER_SCRIPT
            if (docker_image is None) and (docker_container_id is None)
            else os.path.join(root_dir, "run_jupyter_cell.py")
        )

        try:
            execute_fn = (
                execute_python_cmd_docker
                if (docker_image is not None) or (docker_container_id is not None)
                else execute_python_cmd_local
            )
            # In local mode, pass absolute path to the buffer file and run with cwd=root_dir
            code_file_arg = (
                (filename if (docker_image is None and docker_container_id is None) else base_filename)
            )
            result, _ = execute_fn(
                command=f"python {script_path} {code_file_arg} {CELL_SEPARATOR}",
                docker_container_id=docker_container_id,
                docker_image=docker_image,
                session_timeout=session_timeout,
                root_dir=root_dir,
                input_filenames=[filename],
            )
        except Exception as e:
            raise e

        return result

    # first, try without unescaping newlines
    result = _run_cell_code(cell_code)
    if result == "":  # if the cell failed, try unescaping newlines
        # unescape most newlines
        cell_code = re.sub(r"\\{1,}n", r"\n", cell_code)  # \\\\n -> \n
        result = _run_cell_code(cell_code)
    return result
