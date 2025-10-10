import ast
import sys
import os
import builtins
import shutil
from contextlib import redirect_stdout, redirect_stderr


def _is_under(path: str, root: str) -> bool:
    try:
        path_real = os.path.realpath(os.path.abspath(path))
        root_real = os.path.realpath(os.path.abspath(root))
        return os.path.commonpath([path_real, root_real]) == root_real
    except Exception:
        return False


def _install_readonly_guard():
    readonly_dirs = os.environ.get("READONLY_DIRS", "").split(":")
    readonly_dirs = [d for d in readonly_dirs if d]
    if not readonly_dirs:
        return
    def _deny_if_protected(path: str):
        for d in readonly_dirs:
            if _is_under(path, d):
                raise PermissionError(f"Write operation denied for protected path: {path}")

    original_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        # Detect write-like modes
        if any(m in mode for m in ("w", "a", "+", "x")):
            _deny_if_protected(file if isinstance(file, str) else str(file))
        return original_open(file, mode, *args, **kwargs)

    def guarded_remove(path):
        _deny_if_protected(path)
        return os.remove(path)

    def guarded_rename(src, dst, *args, **kwargs):
        _deny_if_protected(src)
        _deny_if_protected(dst)
        return os.rename(src, dst, *args, **kwargs)

    def guarded_replace(src, dst, *args, **kwargs):
        _deny_if_protected(src)
        _deny_if_protected(dst)
        return os.replace(src, dst, *args, **kwargs)

    def guarded_rmdir(path):
        _deny_if_protected(path)
        return os.rmdir(path)

    def guarded_rmtree(path, *args, **kwargs):
        _deny_if_protected(path)
        return shutil.rmtree(path, *args, **kwargs)

    def guarded_move(src, dst, *args, **kwargs):
        _deny_if_protected(src)
        _deny_if_protected(dst)
        return shutil.move(src, dst, *args, **kwargs)

    builtins.open = guarded_open
    os.remove = guarded_remove
    os.rename = guarded_rename
    os.replace = guarded_replace
    os.rmdir = guarded_rmdir
    shutil.rmtree = guarded_rmtree
    shutil.move = guarded_move

def execute_cell(cell_code, env, CODE_FILE, print_last_expr=False, suppress_output=False):
    cell_code = cell_code.strip()
    if not cell_code:
        return
    try:
        tree = ast.parse(cell_code, mode='exec')
    except Exception as e:
        if not suppress_output:
            print(f'Error parsing code: {e}', file=sys.stderr)
        return
    if not tree.body:
        return
    *body, last = tree.body
    try:
        if body:
            exec(compile(ast.Module(body=body, type_ignores=[]), CODE_FILE, 'exec'), env)
        if print_last_expr and isinstance(last, ast.Expr):
            result = eval(compile(ast.Expression(last.value), CODE_FILE, 'eval'), env)
            if result is not None:
                print(result)
        else:
            exec(compile(ast.Module(body=[last], type_ignores=[]), CODE_FILE, 'exec'), env)
    except Exception:
        if not suppress_output:
            import traceback
            traceback.print_exc()

def main():
    # Install read-only guard for protected directories
    _install_readonly_guard()
    CODE_FILE = sys.argv[1]
    CELL_SEPARATOR = sys.argv[2]
    with open(CODE_FILE, 'r') as f:
        code = f.read()
    cells = code.split(CELL_SEPARATOR)
    cells = [cell.strip() for cell in cells if cell.strip()]
    env = {}
    try:
        import pandas as pd
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.width', 500)
        pd.set_option('display.max_rows', 25)
        env['pd'] = pd
    except ImportError:
        pass  # If pandas isn't used, ignore
    for i, cell in enumerate(cells):
        is_last = (i == len(cells) - 1)
        if is_last:
            execute_cell(cell, env, CODE_FILE, print_last_expr=True, suppress_output=False)
        else:
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    execute_cell(cell, env, CODE_FILE, print_last_expr=False, suppress_output=True)

if __name__ == '__main__':
    main() 