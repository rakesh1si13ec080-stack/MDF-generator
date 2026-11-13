"""
MDF Signal Editor - offline tool to modify / add / delete signals in MDF files.

GUI:  python mdf_signal_editor.py
CLI:  python mdf_signal_editor.py in.mdf -o out.mdf --add "CM.TC_INDICATOR_FLAG.FLAG_0" \
          --points "0:0;10:1;58:0" --ref "CM.TC_INDICATOR_FLAG.FLAG_1"
      python mdf_signal_editor.py in.mdf --list

Points format (time in seconds, value):
    0:0; 10:1; 58:0            (time:value separated by ; , or new lines)
    or one point per line:  "10 1", "10,1" or "10<TAB>1"  (paste from Excel works)

Step mode : each value is held until the next time point (default).
Linear    : values are interpolated linearly between the points.
The original file is never changed; a new file is written.
"""

__author__ = "Rakesh Kyathsandra Jayaramu(uih15068)"

import argparse
import os
import re
import sys

import numpy as np
from asammdf import MDF, Signal

APP_NAME = "MDF Signal Editor"
APP_VERSION = "1.1"

DTYPES = ["auto", "uint8", "int8", "uint16", "int16", "uint32", "int32",
          "float32", "float64"]


# --------------------------------------------------------------------------
# Core logic (no GUI) - also used by the command line mode
# --------------------------------------------------------------------------
def parse_points(text):
    """Parse 'time:value' pairs. Returns a sorted list of (time, value)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("No time/value points given.")
    points = []
    if ":" in text:
        chunks = [c for c in re.split(r"[;,\n]+", text) if c.strip()]
        for c in chunks:
            if ":" not in c:
                raise ValueError(f"Cannot read point '{c.strip()}'. Use time:value.")
            t, v = c.split(":", 1)
            points.append((float(t), float(v)))
    else:
        for line in re.split(r"[;\n]+", text):
            line = line.strip()
            if not line:
                continue
            parts = [p for p in re.split(r"[,\s\t]+", line) if p]
            if len(parts) != 2:
                raise ValueError(f"Cannot read line '{line}'. Expected: time value")
            points.append((float(parts[0]), float(parts[1])))
    points.sort(key=lambda p: p[0])
    times = [p[0] for p in points]
    if len(set(times)) != len(times):
        raise ValueError("The same time is given twice.")
    return points


def build_profile(timestamps, points, mode="step"):
    """Values for every timestamp from the given points."""
    t = np.asarray(timestamps, dtype=float)
    pt = np.array([p[0] for p in points], dtype=float)
    pv = np.array([p[1] for p in points], dtype=float)
    if mode == "linear":
        return np.interp(t, pt, pv)
    idx = np.searchsorted(pt, t, side="right") - 1
    idx[idx < 0] = 0  # before the first point -> first value
    return pv[idx]


def choose_dtype(values, requested="auto", like=None):
    if requested and requested != "auto":
        return np.dtype(requested)
    if like is not None and np.issubdtype(like, np.number):
        if np.issubdtype(like, np.integer) and not np.all(np.mod(values, 1) == 0):
            return np.dtype("float64")
        if np.issubdtype(like, np.integer):
            info = np.iinfo(like)
            if values.min() >= info.min and values.max() <= info.max:
                return like
            return np.dtype("int64")
        return like
    if np.all(np.mod(values, 1) == 0):
        lo, hi = values.min(), values.max()
        for d in ("uint8", "int8", "uint16", "int16", "uint32", "int32"):
            info = np.iinfo(d)
            if lo >= info.min and hi <= info.max:
                return np.dtype(d)
        return np.dtype("int64")
    return np.dtype("float64")


class MdfEditor:
    """Holds a loaded MDF file and a list of pending changes."""

    def __init__(self, path):
        self.path = path
        self.mdf = MDF(path)
        self.changes = []  # list of dicts
        self._signals = None

    # ---- information -------------------------------------------------
    def signals(self):
        if self._signals is not None:
            return self._signals
        out = []
        for gi, grp in enumerate(self.mdf.groups):
            for ci, ch in enumerate(grp.channels):
                if self.mdf.masters_db.get(gi) == ci:
                    continue  # time channel of the group
                try:
                    s = self.mdf.get(group=gi, index=ci)
                except Exception:
                    continue
                t = s.timestamps
                info = {"name": ch.name, "group": gi, "index": ci,
                        "unit": s.unit or "", "samples": len(t),
                        "start": float(t[0]) if len(t) else 0.0,
                        "end": float(t[-1]) if len(t) else 0.0,
                        "rate_ms": float(np.median(np.diff(t)) * 1000) if len(t) > 1 else 0.0,
                        "min": "", "max": ""}
                if len(t) and np.issubdtype(s.samples.dtype, np.number):
                    info["min"] = float(np.min(s.samples))
                    info["max"] = float(np.max(s.samples))
                out.append(info)
        self._signals = out
        return out

    def names(self):
        return [s["name"] for s in self.signals()]

    def find(self, name):
        for s in self.signals():
            if s["name"] == name:
                return s
        raise KeyError(f"Signal '{name}' not found in the file.")

    def get(self, name, raw=False):
        s = self.find(name)
        return self.mdf.get(group=s["group"], index=s["index"], raw=raw)

    def duration(self):
        sig = self.signals()
        if not sig:
            return 0.0, 0.0
        return min(s["start"] for s in sig), max(s["end"] for s in sig)

    # ---- building the new samples ------------------------------------
    def new_samples(self, change):
        """Return (timestamps, new_values, original_values_or_None)."""
        kind = change["type"]
        if kind == "modify":
            orig = self.get(change["name"])
            t = orig.timestamps
            if not np.issubdtype(orig.samples.dtype, np.number):
                orig = self.get(change["name"], raw=True)
            vals = build_profile(t, change["points"], change.get("mode", "step"))
            base = orig.samples.astype(float)
            w = change.get("window")
            if w:
                mask = (t >= w[0]) & (t < w[1])
                vals = np.where(mask, vals, base)
            return t, vals, base
        if kind == "add":
            ref = change.get("ref")
            if ref:
                t = self.get(ref).timestamps
            else:
                start, end = self.duration()
                step = float(change.get("rate_ms") or 10) / 1000.0
                t = np.round(np.arange(start, end + step / 2, step), 9)
            vals = build_profile(t, change["points"], change.get("mode", "step"))
            return t, vals, None
        raise ValueError(kind)

    # ---- changes -----------------------------------------------------
    def validate(self, change):
        names = self.names()
        pending_add = [c["name"] for c in self.changes if c["type"] == "add"]
        if change["type"] in ("modify", "delete") and change["name"] not in names:
            raise ValueError(f"Signal '{change['name']}' not found in the file.")
        if change["type"] == "add":
            if not change["name"].strip():
                raise ValueError("Please enter a name for the new signal.")
            if change["name"] in names or change["name"] in pending_add:
                raise ValueError(f"A signal named '{change['name']}' already exists. "
                                 "Use 'Modify' instead, or pick another name.")
            if change.get("ref") and change["ref"] not in names:
                raise ValueError(f"Time base signal '{change['ref']}' not found.")
        if change["type"] in ("modify", "add"):
            if not change.get("points"):
                raise ValueError("No time/value points given.")

    def add_change(self, change):
        self.validate(change)
        # a new modify/delete of the same signal replaces the old one
        self.changes = [c for c in self.changes
                        if not (c["name"] == change["name"] and c["type"] == change["type"])]
        self.changes.append(change)

    @staticmethod
    def describe(change):
        pts = "; ".join(f"{t:g}s={v:g}" for t, v in change.get("points", []))
        if change["type"] == "delete":
            return f"DELETE  {change['name']}"
        extra = f" [{change.get('mode', 'step')}]"
        if change["type"] == "modify":
            w = change.get("window")
            if w:
                extra += f" only {w[0]:g}-{w[1]:g}s"
            return f"MODIFY  {change['name']}: {pts}{extra}"
        tb = f"time base of {change['ref']}" if change.get("ref") else f"{change.get('rate_ms', 10)} ms"
        return f"ADD     {change['name']}: {pts}{extra} ({tb})"

    # ---- write -------------------------------------------------------
    def save(self, out_path, changes=None):
        changes = self.changes if changes is None else changes
        if os.path.abspath(out_path) == os.path.abspath(self.path):
            raise ValueError("Please save to a new file name; the original stays unchanged.")
        mods = {c["name"]: c for c in changes if c["type"] == "modify"}
        dels = {c["name"] for c in changes if c["type"] == "delete"}
        adds = [c for c in changes if c["type"] == "add"]
        adds_by_ref = {}
        free_adds = []
        for c in adds:
            if c.get("ref"):
                adds_by_ref.setdefault(c["ref"], []).append(c)
            else:
                free_adds.append(c)

        new = MDF(version=self.mdf.version)
        try:
            new.header.start_time = self.mdf.header.start_time
        except Exception:
            pass
        try:
            new.header.comment = self.mdf.header.comment
        except Exception:
            pass

        sig_by_pos = {(s["group"], s["index"]): s for s in self.signals()}
        for gi, grp in enumerate(self.mdf.groups):
            out_sigs = []
            for ci, ch in enumerate(grp.channels):
                info = sig_by_pos.get((gi, ci))
                if info is None:
                    continue
                name = info["name"]
                sig = self.mdf.get(group=gi, index=ci, raw=True)
                if name in mods:
                    phys = self.mdf.get(group=gi, index=ci)
                    t, vals, _ = self.new_samples(mods[name])
                    if np.issubdtype(phys.samples.dtype, np.number):
                        # work in physical values, store without conversion
                        dt = choose_dtype(vals, mods[name].get("dtype", "auto"), phys.samples.dtype)
                        sig = Signal(vals.astype(dt), t, name=name, unit=phys.unit,
                                     comment=phys.comment)
                    else:
                        # text table: user entered raw values, keep conversion
                        dt = choose_dtype(vals, "auto", sig.samples.dtype)
                        sig = Signal(vals.astype(dt), t, name=name, unit=sig.unit,
                                     comment=sig.comment, conversion=sig.conversion,
                                     raw=True)
                if name not in dels:
                    out_sigs.append(sig)
                for c in adds_by_ref.get(name, []):
                    t, vals, _ = self.new_samples(c)
                    dt = choose_dtype(vals, c.get("dtype", "auto"))
                    out_sigs.append(Signal(vals.astype(dt), t, name=c["name"],
                                           unit=c.get("unit", ""),
                                           comment=c.get("comment", "Added with MDF Signal Editor")))
            if out_sigs:
                new.append(out_sigs, common_timebase=True)
        for c in free_adds:
            t, vals, _ = self.new_samples(c)
            dt = choose_dtype(vals, c.get("dtype", "auto"))
            new.append([Signal(vals.astype(dt), t, name=c["name"], unit=c.get("unit", ""),
                               comment=c.get("comment", "Added with MDF Signal Editor"))],
                       common_timebase=True)
        new.save(out_path, overwrite=True)
        new.close()
        return out_path


# --------------------------------------------------------------------------
# Signal database (CAN.py) - names offered in the "Add new signal" drop-down
# --------------------------------------------------------------------------
SIGNAL_DB_FILE = "CAN.py"


def app_dir():
    """Folder of the script (or of the .exe when frozen with PyInstaller)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def full_signal_name(name, prefix):
    """'Conti_Veh_CAN.YawRate.YawRate' + prefix 'MTS.ADAS_CAN.Conti_Veh_CAN'
    -> 'MTS.ADAS_CAN.Conti_Veh_CAN.YawRate.YawRate' (as written in the measurement)."""
    if not prefix:
        return name
    first, _, rest = name.partition(".")
    if rest and prefix.split(".")[-1] == first:
        return f"{prefix}.{rest}"
    return f"{prefix}.{name}"


def load_signal_db(path):
    """Read signal names from a generated CAN.py (or a .txt with one name per line).

    CAN.py is read as text (with ast), not imported, so its own imports
    (utils.signal_database_handler ...) are not needed.
    """
    import ast
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    names = []
    if path.lower().endswith(".py"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            lists = {}  # list variable name -> signal names
            for node in tree.body:
                if not isinstance(node, ast.Assign) or not isinstance(node.value, (ast.List, ast.Tuple)):
                    continue
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if not any(t.endswith("_SIGNALS") for t in targets):
                    continue
                items = [el.value.strip() for el in node.value.elts
                         if isinstance(el, ast.Constant) and isinstance(el.value, str)]
                for t in targets:
                    lists[t] = items
            # prefixes from: build_tree(X__SIGNALS, prefix='MTS.ADAS_CAN.Conti_Veh_CAN')
            prefixes = {}
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "build_tree"):
                    continue
                lst = node.args[0].id if node.args and isinstance(node.args[0], ast.Name) else None
                pre = None
                for kw in node.keywords:
                    if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                        pre = kw.value.value
                if pre is None and len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    pre = node.args[1].value
                if lst and pre:
                    prefixes[lst] = pre
            for lst, items in lists.items():
                names.extend(full_signal_name(n, prefixes.get(lst)) for n in items)
        if not names:  # fallback: any quoted dotted name
            names = re.findall(r"""['"]([A-Za-z_]\w*(?:\.\w+)+)['"]""", text)
    else:
        names = [ln.strip() for ln in text.splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
    return sorted(set(n for n in names if n), key=str.lower)


def search_names(names, query, limit=None):
    """Case-insensitive search; every word of the query must appear in the name."""
    words = (query or "").lower().split()
    if not words:
        out = list(names)
    else:
        out = [n for n in names if all(w in n.lower() for w in words)]
    return out[:limit] if limit else out


def default_output_name(path):
    base, ext = os.path.splitext(path)
    return f"{base}_modified{ext or '.mdf'}"


# --------------------------------------------------------------------------
# Command line mode (automation)
# --------------------------------------------------------------------------
def run_cli(argv):
    p = argparse.ArgumentParser(description=f"{APP_NAME} (command line mode)")
    p.add_argument("input", help="input .mdf / .mf4 file")
    p.add_argument("-o", "--output", help="output file (default: <input>_modified.mdf)")
    p.add_argument("--list", action="store_true", help="list the signals and exit")
    p.add_argument("--add", metavar="NAME", action="append", default=[],
                   help="add a new signal (use with --points)")
    p.add_argument("--modify", metavar="NAME", action="append", default=[],
                   help="modify an existing signal (use with --points)")
    p.add_argument("--delete", metavar="NAME", action="append", default=[],
                   help="delete a signal")
    p.add_argument("--points", action="append", default=[],
                   help='time:value points, e.g. "0:0;10:1;58:0" (one --points per --add/--modify, in order)')
    p.add_argument("--ref", help="time base signal for added signals (default: first signal's group)")
    p.add_argument("--rate-ms", type=float, default=None, help="fixed sample rate for added signals")
    p.add_argument("--mode", choices=["step", "linear"], default="step")
    p.add_argument("--window", nargs=2, type=float, metavar=("FROM", "TO"),
                   help="for --modify: only change values between FROM and TO seconds")
    p.add_argument("--dtype", choices=DTYPES, default="auto")
    p.add_argument("--unit", default="")
    a = p.parse_args(argv)

    ed = MdfEditor(a.input)
    if a.list:
        print(f"{'Signal':70s} {'Unit':8s} {'Samples':>8s} {'Rate ms':>8s} {'Min':>10s} {'Max':>10s}")
        for s in ed.signals():
            mn = f"{s['min']:g}" if s["min"] != "" else ""
            mx = f"{s['max']:g}" if s["max"] != "" else ""
            print(f"{s['name']:70s} {s['unit']:8s} {s['samples']:8d} {s['rate_ms']:8.2f} {mn:>10s} {mx:>10s}")
        return 0

    needs = a.add + a.modify
    if len(a.points) != len(needs):
        p.error("Give one --points for every --add and --modify (in the same order).")
    pts = iter(a.points)
    for name in a.modify:
        ed.add_change({"type": "modify", "name": name, "points": parse_points(next(pts)),
                       "mode": a.mode, "window": a.window, "dtype": a.dtype})
    for name in a.add:
        ref = a.ref if a.rate_ms is None else None
        if ref is None and a.rate_ms is None:
            ref = ed.names()[0]
        ed.add_change({"type": "add", "name": name, "points": parse_points(next(pts)),
                       "mode": a.mode, "ref": ref, "rate_ms": a.rate_ms,
                       "dtype": a.dtype, "unit": a.unit})
    for name in a.delete:
        ed.add_change({"type": "delete", "name": name})
    if not ed.changes:
        p.error("Nothing to do. Use --add, --modify, --delete or --list.")
    out = a.output or default_output_name(a.input)
    for c in ed.changes:
        print(MdfEditor.describe(c))
    ed.save(out)
    print(f"Saved: {out}")
    return 0


# --------------------------------------------------------------------------
# GUI (tkinter - included with Python on Windows)
# --------------------------------------------------------------------------
def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    try:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        HAVE_MPL = True
    except Exception:
        HAVE_MPL = False

    EXAMPLE = "0 0\n10 1\n58 0"

    class SearchCombo(ttk.Combobox):
        """Editable combobox with a live search list.

        - Type any part of a name (several words = all must match): a list of
          matching names opens under the box. Up/Down + Enter or a click picks one.
        - The arrow button shows the current matches (all names when empty).
        - Any other text can still be typed or pasted - it does not have to be
          in the list.
        """
        MAX_SHOWN = 300

        def __init__(self, master, textvariable, **kw):
            super().__init__(master, textvariable=textvariable, **kw)
            self.var = textvariable
            self.all_names = []
            self.popup = None
            self.lb = None
            self.on_change = None
            self.bind("<KeyRelease>", self._on_key)
            self.bind("<Down>", self._on_down)
            self.bind("<Up>", self._on_up)
            self.bind("<Return>", self._on_return)
            self.bind("<Escape>", lambda e: self.hide())
            self.bind("<FocusOut>", lambda e: self.after(150, self._maybe_hide))
            self.bind("<<ComboboxSelected>>", lambda e: (self.hide(), self._changed()))
            self.bind("<<Paste>>", lambda e: self.after(10, self.update_list))
            self.winfo_toplevel().bind("<Configure>", lambda e: self.hide(), add="+")

        def set_names(self, names):
            self.all_names = list(names)
            self["values"] = self.all_names

        def matches(self):
            return search_names(self.all_names, self.var.get())

        def _changed(self):
            if self.on_change:
                self.on_change()

        # -- popup list ---------------------------------------------------
        def _ensure_popup(self):
            if self.popup is not None:
                return
            self.popup = tk.Toplevel(self)
            self.popup.withdraw()
            self.popup.overrideredirect(True)
            try:
                self.popup.attributes("-topmost", True)
            except Exception:
                pass
            fr = ttk.Frame(self.popup, borderwidth=1, relief="solid")
            fr.pack(fill="both", expand=True)
            self.lb = tk.Listbox(fr, font=("Consolas", 10), activestyle="dotbox",
                                 exportselection=False, takefocus=0, borderwidth=0)
            sb = ttk.Scrollbar(fr, orient="vertical", command=self.lb.yview)
            self.lb.configure(yscrollcommand=sb.set)
            self.lb.pack(side="left", fill="both", expand=True)
            sb.pack(side="left", fill="y")
            self.lb.bind("<ButtonRelease-1>", self._on_click)
            self.lb.bind("<Double-Button-1>", self._on_click)

        def update_list(self):
            if str(self.cget("state")) != "normal" or not self.all_names:
                self.hide()
                return
            found = self.matches()
            self["values"] = found  # arrow button shows the same matches
            self._changed()
            q = self.var.get().strip()
            if not q or not found or (len(found) == 1 and found[0] == q):
                self.hide()
                return
            self._ensure_popup()
            self.lb.delete(0, "end")
            for n in found[:self.MAX_SHOWN]:
                self.lb.insert("end", n)
            if len(found) > self.MAX_SHOWN:
                self.lb.insert("end", f"... {len(found) - self.MAX_SHOWN} more - type more to narrow")
            rows = min(12, self.lb.size())
            self.update_idletasks()
            x, y = self.winfo_rootx(), self.winfo_rooty() + self.winfo_height()
            import tkinter.font as tkfont
            fnt = tkfont.Font(font=self.lb.cget("font"))
            longest = max((fnt.measure(n) for n in found[:self.MAX_SHOWN]), default=0)
            w = max(self.winfo_width(), min(longest + 40, self.winfo_screenwidth() - 20))
            x = max(0, min(x, self.winfo_screenwidth() - w - 4))  # keep on screen
            h = rows * (fnt.metrics("linespace") + 1) + 8
            self.popup.geometry(f"{w}x{h}+{x}+{y}")
            self.popup.deiconify()
            self.popup.lift()

        def visible(self):
            return self.popup is not None and self.popup.winfo_viewable()

        def hide(self):
            if self.popup is not None:
                self.popup.withdraw()

        def _maybe_hide(self):
            try:
                f = self.focus_get()
            except Exception:
                f = None
            if f not in (self, self.lb):
                self.hide()

        def _pick(self, idx):
            if idx is None or idx >= self.lb.size():
                return
            txt = self.lb.get(idx)
            if txt.startswith("... "):
                return
            self.var.set(txt)
            self.icursor("end")
            self.hide()
            self.focus_set()
            self._changed()

        # -- keys ---------------------------------------------------------
        def _on_key(self, e):
            if e.keysym in ("Up", "Down", "Return", "Escape", "Tab", "Shift_L", "Shift_R",
                            "Control_L", "Control_R", "Alt_L", "Alt_R", "Left", "Right"):
                return
            self.update_list()

        def _move(self, step):
            if not self.visible():
                return None
            cur = self.lb.curselection()
            i = (cur[0] + step) if cur else (0 if step > 0 else self.lb.size() - 1)
            i = max(0, min(self.lb.size() - 1, i))
            self.lb.selection_clear(0, "end")
            self.lb.selection_set(i)
            self.lb.activate(i)
            self.lb.see(i)
            return "break"

        def _on_down(self, e):
            return self._move(1)  # None -> normal combobox drop-down opens

        def _on_up(self, e):
            return self._move(-1)

        def _on_return(self, e):
            if self.visible() and self.lb.curselection():
                self._pick(self.lb.curselection()[0])
                return "break"
            self.hide()
            return None

        def _on_click(self, e):
            self._pick(self.lb.nearest(e.y))

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title(f"{APP_NAME} {APP_VERSION}")
            self.geometry("1280x780")
            self.minsize(1000, 640)
            self.ed = None
            self._build()

        # ---------------------------------------------------------- layout
        def _build(self):
            style = ttk.Style(self)
            try:
                style.theme_use("vista" if sys.platform == "win32" else "clam")
            except Exception:
                pass
            style.configure("Big.TButton", padding=(12, 6))

            top = ttk.Frame(self, padding=8)
            top.pack(fill="x")
            ttk.Button(top, text="1. Open MDF file...", style="Big.TButton",
                       command=self.open_file).pack(side="left")
            self.file_lbl = ttk.Label(top, text="No file loaded", foreground="#555")
            self.file_lbl.pack(side="left", padx=12)

            body = ttk.PanedWindow(self, orient="horizontal")
            body.pack(fill="both", expand=True, padx=8)

            # --- left: signal list
            left = ttk.Labelframe(body, text="Signals in file", padding=6)
            body.add(left, weight=3)
            sf = ttk.Frame(left)
            sf.pack(fill="x")
            ttk.Label(sf, text="Search:").pack(side="left")
            self.search = tk.StringVar()
            self.search.trace_add("write", lambda *_: self.fill_tree())
            ttk.Entry(sf, textvariable=self.search).pack(side="left", fill="x", expand=True, padx=4)
            cols = ("unit", "samples", "rate", "start", "end", "min", "max")
            self.tree = ttk.Treeview(left, columns=cols, show="tree headings", selectmode="browse")
            self.tree.heading("#0", text="Signal name")
            self.tree.column("#0", width=380)
            for c, w, title in (("unit", 60, "Unit"), ("samples", 70, "Samples"),
                                ("rate", 65, "Rate ms"), ("start", 60, "Start s"),
                                ("end", 60, "End s"), ("min", 70, "Min"), ("max", 70, "Max")):
                self.tree.heading(c, text=title)
                self.tree.column(c, width=w, anchor="e")
            ysb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=ysb.set)
            self.tree.pack(side="left", fill="both", expand=True, pady=(6, 0))
            ysb.pack(side="left", fill="y", pady=(6, 0))
            self.tree.bind("<<TreeviewSelect>>", self.on_select)

            # --- right: edit panel
            right = ttk.Labelframe(body, text="2. Define a change", padding=8)
            body.add(right, weight=2)

            self.action = tk.StringVar(value="modify")
            af = ttk.Frame(right)
            af.pack(fill="x")
            for val, txt in (("modify", "Modify selected signal"),
                             ("add", "Add new signal"),
                             ("delete", "Delete selected signal")):
                ttk.Radiobutton(af, text=txt, value=val, variable=self.action,
                                command=self.refresh_form).pack(side="left", padx=(0, 10))

            g = ttk.Frame(right)
            g.pack(fill="x", pady=8)
            g.columnconfigure(1, weight=1)
            r = 0
            ttk.Label(g, text="Signal name:").grid(row=r, column=0, sticky="w", pady=2)
            self.name = tk.StringVar()
            self.name_entry = SearchCombo(g, textvariable=self.name)
            self.name_entry.grid(row=r, column=1, sticky="ew", pady=2)
            self.name_entry.on_change = self.update_db_hint
            r += 1
            dbf = ttk.Frame(g)
            dbf.grid(row=r, column=1, sticky="ew")
            self.db_hint = ttk.Label(dbf, text="", foreground="#666")
            self.db_hint.pack(side="left")
            self.db_btn = ttk.Button(dbf, text="Signal list...", command=self.choose_db)
            self.db_btn.pack(side="right")
            r += 1
            self.tb_lbl = ttk.Label(g, text="Time base:")
            self.tb_lbl.grid(row=r, column=0, sticky="w", pady=2)
            tbf = ttk.Frame(g)
            tbf.grid(row=r, column=1, sticky="ew")
            tbf.columnconfigure(1, weight=1)
            self.tb_mode = tk.StringVar(value="ref")
            self.rb_ref = ttk.Radiobutton(tbf, text="Same as signal", value="ref",
                                          variable=self.tb_mode, command=self.refresh_form)
            self.rb_ref.grid(row=0, column=0, sticky="w")
            self.ref = tk.StringVar()
            self.ref_cb = ttk.Combobox(tbf, textvariable=self.ref, state="readonly")
            self.ref_cb.grid(row=0, column=1, sticky="ew", padx=4)
            self.rb_rate = ttk.Radiobutton(tbf, text="Fixed rate (ms)", value="rate",
                                           variable=self.tb_mode, command=self.refresh_form)
            self.rb_rate.grid(row=1, column=0, sticky="w")
            self.rate = tk.StringVar(value="10")
            self.rate_e = ttk.Entry(tbf, textvariable=self.rate, width=8)
            self.rate_e.grid(row=1, column=1, sticky="w", padx=4)
            r += 1
            ttk.Label(g, text="Unit / Data type:").grid(row=r, column=0, sticky="w", pady=2)
            udf = ttk.Frame(g)
            udf.grid(row=r, column=1, sticky="w")
            self.unit = tk.StringVar()
            self.unit_e = ttk.Entry(udf, textvariable=self.unit, width=10)
            self.unit_e.pack(side="left")
            self.dtype = tk.StringVar(value="auto")
            ttk.Combobox(udf, textvariable=self.dtype, values=DTYPES, width=9,
                         state="readonly").pack(side="left", padx=6)
            r += 1
            ttk.Label(g, text="Between points:").grid(row=r, column=0, sticky="w", pady=2)
            mf = ttk.Frame(g)
            mf.grid(row=r, column=1, sticky="w")
            self.mode = tk.StringVar(value="step")
            ttk.Radiobutton(mf, text="Step (hold value)", value="step",
                            variable=self.mode).pack(side="left")
            ttk.Radiobutton(mf, text="Linear ramp", value="linear",
                            variable=self.mode).pack(side="left", padx=8)
            r += 1
            self.win_on = tk.BooleanVar(value=False)
            self.win_cb = ttk.Checkbutton(g, text="Only change from (s):", variable=self.win_on,
                                          command=self.refresh_form)
            self.win_cb.grid(row=r, column=0, sticky="w", pady=2)
            wf = ttk.Frame(g)
            wf.grid(row=r, column=1, sticky="w")
            self.w_from = tk.StringVar(value="0")
            self.w_to = tk.StringVar(value="10")
            self.w_from_e = ttk.Entry(wf, textvariable=self.w_from, width=8)
            self.w_from_e.pack(side="left")
            ttk.Label(wf, text=" to (s): ").pack(side="left")
            self.w_to_e = ttk.Entry(wf, textvariable=self.w_to, width=8)
            self.w_to_e.pack(side="left")
            ttk.Label(wf, text="  (original values kept outside)", foreground="#666").pack(side="left")

            pf = ttk.Labelframe(right, text="Time (s) and value - one per line, or 0:0; 10:1; 58:0",
                                padding=4)
            pf.pack(fill="both", expand=True)
            self.points = tk.Text(pf, height=5, font=("Consolas", 11), undo=True)
            self.points.pack(fill="both", expand=True)
            self.points.insert("1.0", EXAMPLE)
            bf = ttk.Frame(right)
            bf.pack(fill="x", pady=(8, 0), side="bottom", before=pf)
            ttk.Label(right, text="Example (step): 0 until 10 s, 1 from 10 s, back to 0 at 58 s. "
                                  "You can paste two columns from Excel.",
                      foreground="#666", wraplength=460).pack(fill="x", pady=(4, 0),
                                                              side="bottom", before=bf)
            ttk.Button(bf, text="Preview plot", command=self.preview).pack(side="left")
            ttk.Button(bf, text="Add to change list  >>", style="Big.TButton",
                       command=self.add_change).pack(side="right")

            # --- bottom: change list + save
            bottom = ttk.Labelframe(self, text="3. Pending changes", padding=6)
            bottom.pack(fill="x", padx=8, pady=(6, 0), side="bottom", before=body)
            self.chg = tk.Listbox(bottom, height=5, font=("Consolas", 10))
            self.chg.pack(side="left", fill="both", expand=True)
            cb = ttk.Frame(bottom)
            cb.pack(side="left", fill="y", padx=6)
            ttk.Button(cb, text="Remove selected", command=self.remove_change).pack(fill="x")
            ttk.Button(cb, text="Clear all", command=self.clear_changes).pack(fill="x", pady=4)

            sv = ttk.Frame(self, padding=8)
            sv.pack(fill="x", side="bottom", before=bottom)
            ttk.Button(sv, text="4. Generate modified MDF file...", style="Big.TButton",
                       command=self.save).pack(side="right")
            self.status = ttk.Label(sv, text="Open an MDF file to start.")
            self.status.pack(side="left")
            self.db_names = []
            self.db_path = None
            default_db = os.path.join(app_dir(), SIGNAL_DB_FILE)
            if os.path.isfile(default_db):
                self.load_db(default_db, quiet=True)
            self.refresh_form()

        # ---------------------------------------------------------- signal list (CAN.py)
        def load_db(self, path, quiet=False):
            try:
                names = load_signal_db(path)
            except Exception as e:
                if not quiet:
                    messagebox.showerror(APP_NAME, f"Could not read the signal list:\n{e}")
                return
            if not names and not quiet:
                messagebox.showwarning(APP_NAME, "No signal names found in that file.")
                return
            self.db_names = names
            self.db_path = path
            self.name_entry.set_names(names)
            self.update_db_hint()

        def choose_db(self):
            path = filedialog.askopenfilename(
                title="Choose signal list (CAN.py or .txt with one name per line)",
                initialdir=os.path.dirname(self.db_path) if self.db_path else app_dir(),
                filetypes=[("Signal list", "*.py *.txt"), ("All files", "*.*")])
            if path:
                self.load_db(path)

        def update_db_hint(self):
            if self.action.get() != "add":
                self.db_hint.config(text="")
                return
            if not self.db_names:
                self.db_hint.config(text="No CAN.py signal list loaded - type the name.")
                return
            q = self.name.get().strip()
            src = os.path.basename(self.db_path)
            if not q:
                txt = f"{len(self.db_names)} signals from {src} - type to search, or enter any name"
            elif q in self.db_names:
                txt = f"✓ signal from {src}"
            else:
                n = len(search_names(self.db_names, q))
                txt = (f"{n} match(es) in {src}" if n else
                       f"Not in {src} - will be added with the name as typed")
            self.db_hint.config(text=txt)

        # ---------------------------------------------------------- actions
        def set_status(self, txt):
            self.status.config(text=txt)
            self.update_idletasks()

        def open_file(self):
            path = filedialog.askopenfilename(
                title="Open MDF file",
                filetypes=[("MDF files", "*.mdf *.mf4 *.dat *.MDF *.MF4"), ("All files", "*.*")])
            if not path:
                return
            self.config(cursor="watch")
            self.set_status("Loading...")
            try:
                self.ed = MdfEditor(path)
                sigs = self.ed.signals()
            except Exception as e:
                self.config(cursor="")
                messagebox.showerror(APP_NAME, f"Could not open the file:\n{e}")
                self.set_status("Open an MDF file to start.")
                return
            self.config(cursor="")
            start, end = self.ed.duration()
            self.file_lbl.config(
                text=f"{os.path.basename(path)}   |   MDF {self.ed.mdf.version}   |   "
                     f"{len(sigs)} signals   |   {start:g} - {end:g} s", foreground="#000")
            self.ref_cb["values"] = self.ed.names()
            if sigs:
                self.ref.set(sigs[0]["name"])
            self.fill_tree()
            self.clear_changes()
            self.set_status("Select a signal to modify, or choose 'Add new signal'.")

        def fill_tree(self):
            self.tree.delete(*self.tree.get_children())
            if not self.ed:
                return
            q = self.search.get().lower().strip()

            def fmt(x):
                return "" if x == "" else f"{x:g}"
            for s in self.ed.signals():
                if q and q not in s["name"].lower():
                    continue
                self.tree.insert("", "end", iid=s["name"], text=s["name"],
                                 values=(s["unit"], s["samples"], f"{s['rate_ms']:.2f}",
                                         f"{s['start']:g}", f"{s['end']:g}",
                                         fmt(s["min"]), fmt(s["max"])))

        def on_select(self, _=None):
            sel = self.tree.selection()
            if not sel:
                return
            name = sel[0]
            if self.action.get() == "add":
                self.ref.set(name)
            else:
                self.name.set(name)
                self.unit.set(self.ed.find(name)["unit"])

        def refresh_form(self):
            act = self.action.get()
            add = act == "add"
            dele = act == "delete"
            self.name_entry.config(state="normal" if add else "readonly")
            self.name_entry.hide()
            # CAN.py names are only offered when adding a new signal
            self.name_entry["values"] = self.name_entry.matches() if add else []
            self.db_btn.config(state="normal" if add else "disabled")
            for w in (self.rb_ref, self.rb_rate):
                w.config(state="normal" if add else "disabled")
            self.ref_cb.config(state="readonly" if add and self.tb_mode.get() == "ref" else "disabled")
            self.rate_e.config(state="normal" if add and self.tb_mode.get() == "rate" else "disabled")
            self.unit_e.config(state="normal" if add else "disabled")
            self.win_cb.config(state="normal" if act == "modify" else "disabled")
            on = act == "modify" and self.win_on.get()
            self.w_from_e.config(state="normal" if on else "disabled")
            self.w_to_e.config(state="normal" if on else "disabled")
            self.points.config(state="disabled" if dele else "normal",
                               background="#eee" if dele else "white")
            if add:
                if self.tree.selection():
                    self.ref.set(self.tree.selection()[0])
                if not self.name.get() or (self.ed and self.name.get() in self.ed.names()):
                    self.name.set("")
                self.name_entry.focus_set()
            else:
                sel = self.tree.selection()
                self.name.set(sel[0] if sel else "")
            if add:
                self.name_entry["values"] = self.name_entry.matches()
            self.update_db_hint()

        def current_change(self):
            if not self.ed:
                raise ValueError("Please open an MDF file first.")
            act = self.action.get()
            name = self.name.get().strip()
            if act in ("modify", "delete") and not name:
                raise ValueError("Please select a signal in the list on the left.")
            if act == "delete":
                return {"type": "delete", "name": name}
            ch = {"type": act, "name": name,
                  "points": parse_points(self.points.get("1.0", "end")),
                  "mode": self.mode.get(), "dtype": self.dtype.get()}
            if act == "modify" and self.win_on.get():
                a, b = float(self.w_from.get()), float(self.w_to.get())
                if b <= a:
                    raise ValueError("'to' time must be greater than 'from' time.")
                ch["window"] = (a, b)
            if act == "add":
                ch["unit"] = self.unit.get()
                if self.tb_mode.get() == "ref":
                    ch["ref"] = self.ref.get()
                else:
                    rate = float(self.rate.get())
                    if rate <= 0:
                        raise ValueError("Rate must be greater than 0 ms.")
                    ch["ref"] = None
                    ch["rate_ms"] = rate
            return ch

        def add_change(self):
            try:
                ch = self.current_change()
                self.ed.add_change(ch)
            except Exception as e:
                messagebox.showwarning(APP_NAME, str(e))
                return
            self.show_changes()
            self.set_status(f"Added: {MdfEditor.describe(ch)}")

        def show_changes(self):
            self.chg.delete(0, "end")
            for c in self.ed.changes:
                self.chg.insert("end", MdfEditor.describe(c))

        def remove_change(self):
            sel = self.chg.curselection()
            if sel and self.ed:
                del self.ed.changes[sel[0]]
                self.show_changes()

        def clear_changes(self):
            if self.ed:
                self.ed.changes = []
            self.chg.delete(0, "end")

        def preview(self):
            try:
                ch = self.current_change()
                if ch["type"] == "delete":
                    return
                if ch["type"] == "add":
                    self.ed.validate(ch)
                t, new, orig = self.ed.new_samples(ch)
            except Exception as e:
                messagebox.showwarning(APP_NAME, str(e))
                return
            if not HAVE_MPL:
                messagebox.showinfo(APP_NAME, "Install matplotlib to see the plot:\n"
                                              "pip install matplotlib")
                return
            win = tk.Toplevel(self)
            win.title(f"Preview - {ch['name']}")
            win.geometry("900x500")
            fig = Figure(figsize=(9, 4.6), dpi=100)
            ax = fig.add_subplot(111)
            if orig is not None:
                ax.plot(t, orig, color="#999", lw=1.2, label="original", drawstyle="steps-post")
            ax.plot(t, new, color="#d62728", lw=1.6, label="new",
                    drawstyle="steps-post" if ch["mode"] == "step" else "default")
            ax.set_xlabel("time [s]")
            ax.set_title(ch["name"], fontsize=10)
            ax.grid(alpha=.3)
            ax.legend(loc="best")
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=win)
            NavigationToolbar2Tk(canvas, win)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()

        def save(self):
            if not self.ed:
                messagebox.showwarning(APP_NAME, "Please open an MDF file first.")
                return
            if not self.ed.changes:
                messagebox.showwarning(APP_NAME, "The change list is empty. "
                                                 "Use 'Add to change list' first.")
                return
            out = filedialog.asksaveasfilename(
                title="Save modified MDF as", defaultextension=os.path.splitext(self.ed.path)[1] or ".mdf",
                initialfile=os.path.basename(default_output_name(self.ed.path)),
                initialdir=os.path.dirname(self.ed.path),
                filetypes=[("MDF files", "*.mdf *.mf4"), ("All files", "*.*")])
            if not out:
                return
            self.config(cursor="watch")
            self.set_status("Writing file...")
            try:
                self.ed.save(out)
            except Exception as e:
                self.config(cursor="")
                messagebox.showerror(APP_NAME, f"Could not save:\n{e}")
                self.set_status("Save failed.")
                return
            self.config(cursor="")
            self.set_status(f"Saved: {out}")
            messagebox.showinfo(APP_NAME, f"Modified file saved:\n{out}\n\n"
                                          f"{len(self.ed.changes)} change(s) applied. "
                                          "The original file was not changed.")

    App().mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(run_cli(sys.argv[1:]))
    run_gui()
