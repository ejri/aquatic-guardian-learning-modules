import sys, time, nbformat, os
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
for path in sys.argv[1:]:
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    err = None
    try:
        NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=False).execute()
    except CellExecutionError as e:
        err = str(e)[:3000]
    dt = time.time() - t0
    print(f"==== {os.path.basename(path)}  {dt:.1f}s  {'ERROR' if err else 'OK'}")
    if err: print(err)
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code": continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream": print(f"[{i}]", o["text"][:700].rstrip())
            elif o.get("output_type") in ("execute_result","display_data") and "text/plain" in o.get("data", {}) and "Figure" not in o["data"]["text/plain"]:
                print(f"[{i}] =>", o["data"]["text/plain"][:700])
            elif o.get("output_type") == "error": print(f"[{i}] ERR", o["ename"], o["evalue"])
    nbformat.write(nb, os.path.join("executed", os.path.basename(path)))
